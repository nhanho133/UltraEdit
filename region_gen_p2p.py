"""
Region-based Image Editing — UltraEdit Paper Pipeline (Eq. 3)
==============================================================

Implements the FULL 4-phase pipeline from the UltraEdit paper:
  Phase 1: Text Processing (caption truncation + P2P edit detection)
  Phase 2: Masking (Mf + Mb + s already computed by create_soft_mask.py)
  Phase 3: Modified Diffusion Engine (SDEdit + P2P + Alternating Blending)
  Phase 4: Decode + Save

Key insight: Prompt2PromptInpaintPipeline already implements Steps 3.1–3.5
internally. We just need to call it with the CORRECT 3 separate mask inputs:
  - mask_image = Mf  (SAM fine-grained mask)
  - temp_mask  = Mb  (Bounding Box mask)       ← kwargs
  - soft_mask  = s   (float, e.g. 0.5)         ← kwargs

The pipeline internally does (line 1706–1713 of sdxl_p2p_pipeline.py):
  Odd steps:  init_mask[Mb \\ Mf region] = s
              latents = (1 - init_mask) * z_bg + init_mask * D_M(z_t)
  Even steps: latents = D_M(z_t)    [free generation → smooth edges]

Usage:
  # Step 1: Generate masks (if not done already)
  python create_soft_mask.py \\
    --image images/example_images/Test_DOCCi.png \\
    --target_object "ghost graffiti" \\
    --output_dir mask_output

  # Step 2: Run region-based edit
  python region_gen_p2p.py \\
    --image mask_output/input_resized.png \\
    --mf_mask mask_output/Mf_fine_grained.png \\
    --mb_mask mask_output/Mb_bounding_box.png \\
    --soft_mask_value 0.5 \\
    --source_caption "a wall with ghost graffiti" \\
    --target_caption "a clean wall without graffiti" \\
    --output_dir region_output \\
    --device cpu
"""

import sys
import os
import argparse
import random
import warnings
import logging
import torch
import numpy as np
from PIL import Image, ImageDraw

# ── Silence noisy warnings ───────────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

_orig_warning = logging.Logger.warning
def _filtered_warning(self, msg, *args, **kwargs):
    if "cross_attention_kwargs" in str(msg) and "not expected" in str(msg):
        return
    _orig_warning(self, msg, *args, **kwargs)
logging.Logger.warning = _filtered_warning

# ── Repo path ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "diffusers", "src"))   # forked diffusers
sys.path.insert(0, os.path.join(ROOT, "data_generation"))
sys.path.insert(0, os.path.join(ROOT, "Long-CLIP"))          # Long-CLIP repo

from sdxl_p2p_pipeline import Prompt2PromptInpaintPipeline


# ── Custom Text Encoders Loader ──────────────────────────────────────────────
def load_custom_encoders(longclip_path: str, bigG_path: str, device: str):
    """
    Load Long-CLIP (ViT-L) and OpenCLIP (ViT-bigG-14).
    Both support 248 tokens context length.
    """
    import torch.nn as nn
    from model import longclip
    from open_clip_long import factory as open_clip

    print(f"\n[Phase 0] Loading Custom Text Encoders...")
    
    # 1. Load Long-CLIP (for text_encoder 1)
    print(f"  > Loading Long-CLIP (ViT-L) from: {longclip_path}")
    vitl_model, _ = longclip.load(longclip_path, device=device)
    vitl_model.eval()
    vitL_encode_fn = vitl_model.encode_text_full
    vitL_tokenize_fn = longclip.tokenize

    # 2. Load OpenCLIP bigG (for text_encoder 2)
    print(f"  > Loading OpenCLIP (ViT-bigG-14) from: {bigG_path} (LAION weights)")
    bigG_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-bigG-14', pretrained=bigG_path
    )
    
    # KPS pos-embedding interpolation (from encode_prompt.py)
    positional_embedding_pre = bigG_model.positional_embedding       
    length, dim = positional_embedding_pre.shape
    keep_len = 20
    new_pos = torch.zeros([4*length-3*keep_len, dim], dtype=positional_embedding_pre.dtype)
    for i in range(keep_len):
        new_pos[i] = positional_embedding_pre[i]
    for i in range(length-1-keep_len):
        new_pos[4*i + keep_len] = positional_embedding_pre[i + keep_len]
        new_pos[4*i + 1 + keep_len] = 3*positional_embedding_pre[i + keep_len]/4 + 1*positional_embedding_pre[i+1+keep_len]/4
        new_pos[4*i + 2+keep_len] = 2*positional_embedding_pre[i+keep_len]/4 + 2*positional_embedding_pre[i+1+keep_len]/4
        new_pos[4*i + 3+keep_len] = 1*positional_embedding_pre[i+keep_len]/4 + 3*positional_embedding_pre[i+1+keep_len]/4

    new_pos[4*length -3*keep_len - 4] = positional_embedding_pre[length-1] + 0*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    new_pos[4*length -3*keep_len - 3] = positional_embedding_pre[length-1] + 1*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    new_pos[4*length -3*keep_len - 2] = positional_embedding_pre[length-1] + 2*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    new_pos[4*length -3*keep_len - 1] = positional_embedding_pre[length-1] + 3*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    bigG_model.positional_embedding = nn.Parameter(new_pos)

    bigG_model.eval().to(device)
    bigG_encode_fn = bigG_model.encode_text_full
    bigG_tokenize_fn = open_clip.get_tokenizer('ViT-bigG-14')

    print("  > Custom encoders loaded OK (context_length=248)")
    return vitL_encode_fn, vitL_tokenize_fn, bigG_encode_fn, bigG_tokenize_fn


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — TEXT PROCESSING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def load_and_resize(image_path: str, target_size: int = 512) -> Image.Image:
    """Load ảnh và center-crop về target_size × target_size."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w > h:
        img = img.resize((int(target_size * w / h), target_size), Image.LANCZOS)
    else:
        img = img.resize((target_size, int(target_size * h / w)), Image.LANCZOS)
    w, h = img.size
    left = (w - target_size) // 2
    top  = (h - target_size) // 2
    return img.crop((left, top, left + target_size, top + target_size))


def safe_truncate_caption(text: str, tokenizer, max_tokens: int = 75) -> str:
    """
    Truncate caption bằng tokenizer thực — đếm tokens, không đếm words.
    SDXL tokenizer: max_length=77 (BOS + 75 content + EOS).
    Long-CLIP/OpenCLIP tokenizer: max_length=248 → max_tokens=246.
    """
    # Custom tokenizers return tensor directly or have different API
    if callable(tokenizer) and not hasattr(tokenizer, 'encode'):
        return text   # Custom tokenizer handles length natively
        
    try:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
        print(f"  [WARN] Caption truncated: {len(tokens)} → {max_tokens} tokens")
        return truncated
    except AttributeError:
        # Fallback if tokenizer doesn't have encode/decode
        return text


def compare_prompts(p1: str, p2: str):
    """Tìm từ khác nhau giữa 2 prompts."""
    w1, w2 = p1.split(), p2.split()
    return [b for a, b in zip(w1, w2) if a != b]


def build_cross_attention_kwargs(src: str, tgt: str,
                                  p2p_threshold: float) -> dict:
    """
    Auto-detect P2P edit type:
      replace: cùng số từ, có từ khác nhau (swap attention)
      refine:  khác số từ (alignment-based injection)
    """
    diff = compare_prompts(src, tgt)
    if len(diff) > 0 and len(src.split()) == len(tgt.split()):
        edit_type = "replace"
        key = " ".join(diff) if len(diff) <= 5 else None
        n_cross = (
            {"default_": 1.0, key: p2p_threshold}
            if key else p2p_threshold
        )
    else:
        edit_type = "refine"
        n_cross = p2p_threshold

    print(f"  P2P edit_type: {edit_type}  |  threshold: {p2p_threshold}")
    print(f"  diff words: {diff[:5]}{'...' if len(diff) > 5 else ''}")
    return {
        "edit_type":       edit_type,
        "n_self_replace":  p2p_threshold,
        "n_cross_replace": n_cross,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SAVE UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def save_comparison(source, mf_mask, mb_mask, src_recon, target,
                    output_dir, prefix):
    """Lưu 5 ảnh side-by-side: Source | Mf | Mb | SourceRecon | Target."""
    w, h = source.size
    n = 5
    comp = Image.new("RGB", (w * n, h), color=(20, 20, 20))
    draw = ImageDraw.Draw(comp)

    items = [
        (source,                    "Source"),
        (mf_mask.convert("RGB"),    "Mf (SAM)"),
        (mb_mask.convert("RGB"),    "Mb (BBox)"),
        (src_recon,                 "Src Recon"),
        (target,                    "Target"),
    ]
    for i, (img, label) in enumerate(items):
        comp.paste(img.resize((w, h)), (i * w, 0))
        draw.rectangle([(i*w+2, 2), (i*w + len(label)*8 + 8, 22)], fill="black")
        draw.text((i*w + 5, 4), label, fill="white")

    path = os.path.join(output_dir, f"{prefix}_comparison.png")
    comp.save(path)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — FULL PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="UltraEdit Region-based Edit (P2P Inpainting + Soft Mask)")

    # ── Inputs ────────────────────────────────────────────────────────────────
    parser.add_argument("--image", required=True,
                        help="Source image (or mask_output/input_resized.png)")
    parser.add_argument("--mf_mask", required=True,
                        help="SAM fine-grained mask (Mf) — white = edit region")
    parser.add_argument("--mb_mask", required=True,
                        help="GroundingDINO bounding box mask (Mb) — white = bbox region")
    parser.add_argument("--source_caption", required=True,
                        help="Ts: Caption mô tả ảnh gốc")
    parser.add_argument("--target_caption", required=True,
                        help="Tt: Caption mô tả ảnh sau edit")

    # ── Mask + Generation params ──────────────────────────────────────────────
    parser.add_argument("--soft_mask_value", type=float, default=0.5,
                        help="s ∈ [0.2, 0.8]: alpha for Mb\\Mf transition zone")
    parser.add_argument("--p2p_threshold", type=float, default=0.7,
                        help="P2P attention injection threshold [0,1]")
    parser.add_argument("--steps", type=int, default=4,
                        help="Denoising steps (SDXL-Turbo: 3-7)")
    parser.add_argument("--strength", type=float, default=0.8,
                        help="SDEdit noise strength [0,1] — how much of z0 to corrupt")
    parser.add_argument("--guidance_scale", type=float, default=0.0,
                        help="CFG scale (SDXL-Turbo: 0.0 = no CFG)")
    parser.add_argument("--seed", type=int, default=42)

    # ── System ────────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", default="region_output")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pipeline_ckpt", default="stabilityai/sdxl-turbo")
    
    # ── Custom Text Encoders ──────────────────────────────────────────────────
    parser.add_argument("--longclip_ckpt", default=None,
                        help="Path to longclip-L.pt (Replaces text_encoder 1). Extends SDXL to 248 tokens.")
    parser.add_argument("--bigg_ckpt", default=None,
                        help="Path to OpenCLIP ViT-bigG-14 bin (Replaces text_encoder 2).")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = torch.float16 if device == "cuda" else torch.float32

    use_custom_encoders = args.longclip_ckpt and args.bigg_ckpt
    
    print(f"\n{'='*60}")
    print(f"  UltraEdit Region-based Pipeline (Paper Eq. 3)")
    print(f"  Device: {device} | Dtype: {dtype}")
    print(f"  Steps: {args.steps} | Strength: {args.strength}")
    print(f"  Soft mask s: {args.soft_mask_value}")
    print(f"  Encoders: {'Long-CLIP + open_clip_long (248 tokens)' if use_custom_encoders else 'Standard CLIP (77 tokens)'}")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3.0: Load Pipeline
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[Phase 3.0] Loading SDXL-Turbo pipeline...")
    pipe = Prompt2PromptInpaintPipeline.from_pretrained(
        args.pipeline_ckpt,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    ).to(device)
    pipe.unet.config.addition_embed_type = None
    pipe.set_progress_bar_config(disable=False)
    print(f"  Pipeline loaded: {args.pipeline_ckpt}")
    print(f"  UNet channels: {pipe.unet.config.in_channels}")

    # ── Load Custom Encoders (if paths provided) ────────────────────────────
    vitL_enc, vitL_tok, bigG_enc, bigG_tok = None, None, None, None
    if use_custom_encoders:
        vitL_enc, vitL_tok, bigG_enc, bigG_tok = load_custom_encoders(
            args.longclip_ckpt, args.bigg_ckpt, device
        )

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: TEXT PROCESSING
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[Phase 1] Text Processing...")
    
    check_tokenizer = vitL_tok if use_custom_encoders else pipe.tokenizer
    src_caption = safe_truncate_caption(args.source_caption, check_tokenizer)
    tgt_caption = safe_truncate_caption(args.target_caption, check_tokenizer)
    print(f"  Ts: \"{src_caption[:80]}{'...' if len(src_caption)>80 else ''}\"")
    print(f"  Tt: \"{tgt_caption[:80]}{'...' if len(tgt_caption)>80 else ''}\"")

    cross_attention_kwargs = build_cross_attention_kwargs(
        src_caption, tgt_caption, args.p2p_threshold)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: LOAD MASKS (already computed by create_soft_mask.py)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[Phase 2] Loading image + masks...")
    image = load_and_resize(args.image, args.image_size)
    image.save(os.path.join(args.output_dir, "input_resized.png"))

    # Mf: SAM fine-grained mask (white = object core)
    mf_mask = Image.open(args.mf_mask).convert("L").resize(
        (args.image_size, args.image_size), Image.NEAREST)

    # Mb: GroundingDINO bounding box mask (white = bbox, includes Mf area)
    mb_mask = Image.open(args.mb_mask).convert("L").resize(
        (args.image_size, args.image_size), Image.NEAREST)

    mf_np = np.array(mf_mask).astype(np.float32) / 255.0
    mb_np = np.array(mb_mask).astype(np.float32) / 255.0

    print(f"  Image:   {image.size}")
    print(f"  Mf mask: core region = {(mf_np > 0.5).mean()*100:.1f}%")
    print(f"  Mb mask: bbox region = {(mb_np > 0.5).mean()*100:.1f}%")
    print(f"  Mb \\ Mf: transition  = {((mb_np > 0.5) & (mf_np < 0.5)).mean()*100:.1f}%")
    print(f"  s = {args.soft_mask_value}")

    # Save mask inputs for debugging
    mf_mask.save(os.path.join(args.output_dir, "input_Mf.png"))
    mb_mask.save(os.path.join(args.output_dir, "input_Mb.png"))

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: MODIFIED DIFFUSION ENGINE
    #   Pipeline internally handles:
    #     3.1: SDEdit init   (VAE.encode → z0 → add_noise → zT)
    #     3.2: P2P caching   (source caption cross-attn maps stored)
    #     3.3: Denoising loop (target caption + P2P inject)
    #     3.4: Alternating blending (Eq. 3)
    #     3.5: VAE decode
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[Phase 3] Running Modified Diffusion Engine...")
    print(f"  3.1: SDEdit init (strength={args.strength})")
    print(f"  3.2: P2P Attention Caching (Ts → cross-attn maps)")
    print(f"  3.3: Controlled Denoising ({args.steps} steps, Tt + P2P inject)")
    print(f"  3.4: Alternating Blending:")
    print(f"        Odd steps:  z_{{t-1}} = (1-Ms)*z_bg + Ms*D_M(z_t)")
    print(f"        Even steps: z_{{t-1}} = D_M(z_t)  [free → smooth edges]")
    print(f"        where Ms: Mf=1.0, Mb\\Mf={args.soft_mask_value}, outside=0.0")
    print(f"  3.5: VAE decode → target image")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Monkey-patch Encoders ───────────────────────────────────────────────
    old_te1, old_tok1 = None, None
    old_te2, old_tok2 = None, None
    
    if use_custom_encoders and vitL_enc is not None:
        old_te1, old_tok1 = pipe.text_encoder, pipe.tokenizer
        old_te2, old_tok2 = pipe.text_encoder_2, pipe.tokenizer_2
        
        pipe.text_encoder = vitL_enc
        pipe.tokenizer    = vitL_tok
        pipe.text_encoder_2 = bigG_enc
        pipe.tokenizer_2  = bigG_tok
        print("  [Custom Encoders] Monkey-patched BOTH text encoders (248 tokens)")

    # THE CORRECT CALL — 3 separate mask arguments:
    #   mask_image = Mf  (fine-grained SAM mask)
    #   temp_mask  = Mb  (bounding box mask)         ← kwargs
    #   soft_mask  = s   (float transition value)     ← kwargs
    out = pipe(
        prompt=[src_caption, tgt_caption],
        image=image,
        mask_image=mf_mask,                           # Mf (SAM)
        temp_mask=mb_mask,                            # Mb (BBox) → kwargs
        soft_mask=args.soft_mask_value,               # s (float) → kwargs
        num_inference_steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        cross_attention_kwargs=cross_attention_kwargs,
        output_type="pil",
    ).images
    # out[0] = reconstruction from source caption (Ts)
    # out[1] = edited image from target caption (Tt) + P2P + mask blend

    # ── Restore Encoders ────────────────────────────────────────────────────
    if old_te1 is not None:
        pipe.text_encoder, pipe.tokenizer = old_te1, old_tok1
        pipe.text_encoder_2, pipe.tokenizer_2 = old_te2, old_tok2

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[Phase 4] Saving results to {args.output_dir}/")
    prefix = f"s{args.soft_mask_value}_p2p{args.p2p_threshold}_seed{args.seed}"

    out[0].save(os.path.join(args.output_dir, f"{prefix}_src_recon.png"))
    out[1].save(os.path.join(args.output_dir, f"{prefix}_target.png"))

    comp_path = save_comparison(
        source=image, mf_mask=mf_mask, mb_mask=mb_mask,
        src_recon=out[0], target=out[1],
        output_dir=args.output_dir, prefix=prefix,
    )

    print(f"  {prefix}_src_recon.png   ← source reconstruction")
    print(f"  {prefix}_target.png      ← TARGET edited image")
    print(f"  {prefix}_comparison.png  ← 5-panel comparison")
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()