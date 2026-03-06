"""
Region-based Image Editing — UltraEdit Pipeline (Final Stable Version)
=======================================================================
Dùng Prompt2PromptInpaintPipeline từ repo UltraEdit.

Pipeline đúng theo paper (Eq. 3):
  3.1  SDEdit init:        VAE.encode(source) → z0 → add_noise → zT
  3.2  P2P cache:          UNet(Ts) → lưu cross-attention maps
  3.3  Denoising loop:     UNet(Tt) + P2P inject attention maps
  3.4  Alternating blend:
         Odd steps:  z = (1 - Ms) * z_bg  +  Ms * D_M(z_t)
         Even steps: z = D_M(z_t)   [free → smooth edges]
         Ms: Mf=1.0,  Mb\Mf=soft_mask_value,  ngoài=0.0
  3.5  VAE decode → target image

Token limit:
  Standard:  77 tokens (CLIP)
  --use_long_clip: 248 tokens (Long-CLIP + OpenCLIP bigG)
    → Pre-compute prompt_embeds rồi truyền trực tiếp vào pipe()
    → KHÔNG monkey-patch text_encoder (tránh crash)
    → pipeline.encode_prompt() bị bypass hoàn toàn

Usage:
  # Standard 77 tokens:
  python region_gen_p2p.py \
    --image mask_output/input_resized.png \
    --mf_mask mask_output/Mf_fine_grained.png \
    --mb_mask mask_output/Mb_bounding_box.png \
    --source_caption "white monster face graffiti on brick wall" \
    --target_caption "blue monster face graffiti on brick wall" \
    --device cpu

  # Extended 248 tokens:
  python region_gen_p2p.py ... --use_long_clip \
    --source_caption "A zoomed-in view of a very long caption..." \
    --target_caption "A zoomed-in view of another long caption..."
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

# ── Silence noisy warnings ────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
_orig_warning = logging.Logger.warning
def _filtered_warning(self, msg, *args, **kwargs):
    if "cross_attention_kwargs" in str(msg) and "not expected" in str(msg):
        return
    _orig_warning(self, msg, *args, **kwargs)
logging.Logger.warning = _filtered_warning

# ── Repo paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "diffusers", "src"))
sys.path.insert(0, os.path.join(ROOT, "data_generation"))
sys.path.insert(0, os.path.join(ROOT, "Long-CLIP"))

from sdxl_p2p_pipeline import Prompt2PromptInpaintPipeline


# ═════════════════════════════════════════════════════════════════════════════
# LONG-CLIP EMBEDDINGS  (248 tokens)
# Cách an toàn: pre-compute embeddings, bypass encode_prompt() hoàn toàn.
# KHÔNG monkey-patch text_encoder — tránh crash do API mismatch.
# ═════════════════════════════════════════════════════════════════════════════

def build_longclip_embeddings(src_caption: str, tgt_caption: str,
                               device: str, dtype) -> dict | None:
    """
    Pre-compute prompt_embeds + pooled_prompt_embeds dùng:
      - Long-CLIP ViT-L      → hidden states (seq, 768)
      - OpenCLIP ViT-bigG-14 → hidden states (seq, 1280) + pooled (1280)

    Concat → prompt_embeds (2, seq, 2048), pooled (2, 1280)
    Truyền trực tiếp vào pipe() → pipeline KHÔNG gọi tokenizer/text_encoder.

    Returns None nếu load thất bại → fallback sang CLIP 77 tokens.
    """
    import torch.nn as nn

    # ── Long-CLIP ViT-L ───────────────────────────────────────────────────────
    print("    [Long-CLIP] Loading ViT-L from HuggingFace Hub...")
    try:
        from model import longclip
        from huggingface_hub import hf_hub_download
        lc_path = hf_hub_download(
            repo_id="BeichenZhang/LongCLIP-L", filename="longclip-L.pt")
        lc_path = os.path.realpath(lc_path)  # resolve symlink → blob path thực
        lc_model, _ = longclip.load(lc_path, device=device)
        lc_model.eval()

        # Verify đây thực sự là Long-CLIP (context_length=248)
        # Nếu load ra CLIP gốc (context_length=77) → file cache bị corrupt
        if not hasattr(lc_model, 'encode_text_full'):
            raise RuntimeError(
                f"encode_text_full không có — model là CLIP gốc "
                f"(context_length={lc_model.context_length}). "
                f"Xóa cache: rm -rf ~/.cache/huggingface/hub/models--BeichenZhang*")
        if lc_model.context_length < 200:
            raise RuntimeError(
                f"context_length={lc_model.context_length} — cần 248. "
                f"Xóa cache: rm -rf ~/.cache/huggingface/hub/models--BeichenZhang*")

        print(f"    Long-CLIP OK  context_length={lc_model.context_length}")
    except Exception as e:
        print(f"    [FAIL] Long-CLIP: {e}")
        return None

    # ── OpenCLIP ViT-bigG-14 ──────────────────────────────────────────────────
    # Cache path: ~/.cache/ultraedit/openclip_vitbigG14.pt
    # Lần đầu: download từ Hub → lưu cache
    # Lần sau: load từ cache local → không download lại
    OPENCLIP_CACHE = os.path.expanduser(
        "~/.cache/ultraedit/openclip_vitbigG14.pt")
    print("    [OpenCLIP] Loading ViT-bigG-14...")
    try:
        from open_clip_long import factory as open_clip
        if os.path.exists(OPENCLIP_CACHE):
            print(f"    Loading from local cache: {OPENCLIP_CACHE}")
            bigG, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14")
            ckpt = torch.load(OPENCLIP_CACHE, map_location=device)
            bigG.load_state_dict(ckpt)
        else:
            print("    First run — downloading from HuggingFace Hub...")
            bigG, _, _ = open_clip.create_model_and_transforms(
                "ViT-bigG-14", pretrained="laion2b_s39b_b160k")
            os.makedirs(os.path.dirname(OPENCLIP_CACHE), exist_ok=True)
            torch.save(bigG.state_dict(), OPENCLIP_CACHE)
            print(f"    Cached to: {OPENCLIP_CACHE}")

        # KPS positional embedding interpolation
        # Giữ 20 tokens đầu, nội suy tuyến tính 4x phần còn lại
        # → context_length tăng từ ~77 lên ~248
        pos   = bigG.positional_embedding.detach()
        L, D  = pos.shape
        keep  = 20
        new_L = 4 * L - 3 * keep
        new_pos = torch.zeros([new_L, D], dtype=pos.dtype)
        # Copy phần giữ nguyên (short-range tokens)
        new_pos[:keep] = pos[:keep]
        # Nội suy phần dài (long-range tokens)
        for i in range(L - 1 - keep):
            k = 4 * i + keep
            new_pos[k]   =  pos[i + keep]
            new_pos[k+1] = (3*pos[i+keep]   +   pos[i+1+keep]) / 4
            new_pos[k+2] = (2*pos[i+keep]   + 2*pos[i+1+keep]) / 4
            new_pos[k+3] = (  pos[i+keep]   + 3*pos[i+1+keep]) / 4
        # Extrapolate 4 tokens cuối
        d = pos[-1] - pos[-2]
        for j in range(4):
            new_pos[new_L - 4 + j] = pos[-1] + j * d / 4
        bigG.positional_embedding = nn.Parameter(new_pos)
        bigG.eval().to(device)
        bigG_tok = open_clip.get_tokenizer("ViT-bigG-14")
        print(f"    OpenCLIP OK  new_ctx={new_L}")
    except Exception as e:
        print(f"    [FAIL] OpenCLIP: {e}")
        return None

    # ── Encode function ───────────────────────────────────────────────────────
    def encode(caption: str):
        with torch.no_grad():
            # Encoder 1: Long-CLIP → (1, seq1, 768)
            lc_tok = longclip.tokenize([caption]).to(device)
            emb1   = lc_model.encode_text_full(lc_tok).to(dtype)

            # Encoder 2: OpenCLIP → (1, seq2, 1280) + pooled EOS (1, 1280)
            bg_tok = bigG_tok([caption]).to(device)
            emb2   = bigG.encode_text_full(bg_tok).to(dtype)
            pooled = emb2[
                torch.arange(emb2.shape[0]),
                bg_tok.argmax(dim=-1)
            ]

            # Pad seq dim nếu 2 encoder cho ra độ dài khác nhau
            s1, s2 = emb1.shape[1], emb2.shape[1]
            if s1 < s2:
                emb1 = torch.cat([emb1,
                    torch.zeros(1, s2-s1, emb1.shape[2],
                                device=device, dtype=dtype)], dim=1)
            elif s2 < s1:
                emb2 = torch.cat([emb2,
                    torch.zeros(1, s1-s2, emb2.shape[2],
                                device=device, dtype=dtype)], dim=1)

            # Concat hidden dim: (1, seq, 768+1280) = (1, seq, 2048)
            pe = torch.cat([emb1, emb2], dim=-1)
        return pe, pooled  # (1, seq, 2048), (1, 1280)

    print("    Encoding captions...")
    src_pe, src_pool = encode(src_caption)
    tgt_pe, tgt_pool = encode(tgt_caption)

    # Batch [src, tgt]: (2, seq, 2048) và (2, 1280)
    prompt_embeds        = torch.cat([src_pe,   tgt_pe],   dim=0)
    pooled_prompt_embeds = torch.cat([src_pool, tgt_pool], dim=0)

    print(f"    prompt_embeds:        {tuple(prompt_embeds.shape)}"
          f"  ({prompt_embeds.shape[1]} tokens)")
    print(f"    pooled_prompt_embeds: {tuple(pooled_prompt_embeds.shape)}")
    return {
        "prompt_embeds":        prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


# ═════════════════════════════════════════════════════════════════════════════
# UTILS
# ═════════════════════════════════════════════════════════════════════════════

def load_and_resize(path: str, size: int = 512) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w > h:
        img = img.resize((int(size * w / h), size), Image.LANCZOS)
    else:
        img = img.resize((size, int(size * h / w)), Image.LANCZOS)
    w, h = img.size
    return img.crop(((w-size)//2, (h-size)//2, (w+size)//2, (h+size)//2))


def safe_truncate(text: str, tokenizer, max_tokens: int = 75) -> str:
    """
    Truncate caption dùng tokenizer thực (đếm tokens, không đếm words).
    Chỉ dùng khi không có Long-CLIP.
    """
    try:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
        print(f"  [WARN] Caption truncated {len(tokens)} → {max_tokens} tokens")
        return truncated
    except Exception:
        return text


def build_cross_attention_kwargs(src: str, tgt: str, threshold: float) -> dict:
    """
    Auto-detect P2P edit type:
      replace: cùng số từ + có từ khác → swap attention map tại từ thay đổi
               VD: "white monster face" → "blue monster face"
      refine:  khác số từ             → sequence-alignment injection
               VD: caption dài khác nhau hoàn toàn
    """
    diff = [b for a, b in zip(src.split(), tgt.split()) if a != b]
    same_len = len(src.split()) == len(tgt.split())

    if len(diff) > 0 and same_len:
        edit_type = "replace"
        key       = " ".join(diff) if len(diff) <= 5 else None
        n_cross   = {"default_": 1.0, key: threshold} if key else threshold
    else:
        edit_type = "refine"
        n_cross   = threshold

    print(f"  P2P edit_type : {edit_type}  |  threshold: {threshold}")
    print(f"  diff words    : {diff[:8]}{'...' if len(diff)>8 else ''}")
    return {
        "edit_type":       edit_type,
        "n_self_replace":  threshold,
        "n_cross_replace": n_cross,
        "prompts":         [src, tgt],
    }


def save_comparison(source, mf_mask, mb_mask, src_recon, target,
                    output_dir: str, prefix: str) -> str:
    """5-panel comparison: Source | Mf | Mb | Src Recon | Target."""
    w, h = source.size
    comp = Image.new("RGB", (w * 5, h), (20, 20, 20))
    draw = ImageDraw.Draw(comp)
    items = [
        (source,                     "Source"),
        (mf_mask.convert("RGB"),     "Mf (SAM)"),
        (mb_mask.convert("RGB"),     "Mb (BBox)"),
        (src_recon,                  "Src Recon"),
        (target,                     "Target"),
    ]
    for i, (img, label) in enumerate(items):
        comp.paste(img.resize((w, h)), (i * w, 0))
        draw.rectangle([(i*w+2, 2), (i*w + len(label)*8 + 8, 22)], fill="black")
        draw.text((i*w+5, 4), label, fill="white")
    path = os.path.join(output_dir, f"{prefix}_comparison.png")
    comp.save(path)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="UltraEdit Region-based Edit — P2P + Soft Mask (Stable)")

    # ── Input paths ────────────────────────────────────────────────────────────
    parser.add_argument("--image",           required=True,
                        help="Source image path")
    parser.add_argument("--mf_mask",         required=True,
                        help="SAM fine-grained mask Mf (white=edit region)")
    parser.add_argument("--mb_mask",         required=True,
                        help="BBox mask Mb (white=bbox region)")
    parser.add_argument("--source_caption",  required=True,
                        help="Ts: caption mô tả ảnh gốc")
    parser.add_argument("--target_caption",  required=True,
                        help="Tt: caption mô tả ảnh sau edit")

    # ── Mask params ────────────────────────────────────────────────────────────
    parser.add_argument("--soft_mask_value", type=float, default=0.5,
                        help="s ∈ [0.2,0.8] — blend factor vùng Mb\\Mf")

    # ── Generation params ──────────────────────────────────────────────────────
    parser.add_argument("--p2p_threshold",   type=float, default=0.7,
                        help="P2P attention injection threshold [0,1]")
    parser.add_argument("--steps",           type=int,   default=4,
                        help="Denoising steps (SDXL-Turbo: 3-7)")
    parser.add_argument("--strength",        type=float, default=0.8,
                        help="SDEdit noise strength [0,1]")
    parser.add_argument("--guidance_scale",  type=float, default=0.0,
                        help="CFG (SDXL-Turbo: 0.0)")
    parser.add_argument("--seed",            type=int,   default=42)

    # ── System ────────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir",      default="region_output")
    parser.add_argument("--image_size",      type=int,   default=512)
    parser.add_argument("--device",          default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pipeline_ckpt",   default="stabilityai/sdxl-turbo")

    # ── Extended token support ─────────────────────────────────────────────────
    parser.add_argument("--use_long_clip",   action="store_true",
                        help="Dùng Long-CLIP + OpenCLIP để extend context 77→248 tokens. "
                             "Tự động download từ HuggingFace Hub. "
                             "KHÔNG monkey-patch encoder — inject embeddings trực tiếp.")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
              if args.device == "auto" else args.device
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"\n{'='*60}")
    print(f"  UltraEdit Region-based Edit (Stable)")
    print(f"  Device   : {device} | Dtype: {dtype}")
    print(f"  Steps    : {args.steps} | Strength: {args.strength}")
    print(f"  soft_s   : {args.soft_mask_value} | p2p: {args.p2p_threshold}")
    print(f"  Encoder  : {'Long-CLIP + OpenCLIP (248 tokens)' if args.use_long_clip else 'CLIP (77 tokens)'}")
    print(f"{'='*60}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── [1] Load Pipeline ─────────────────────────────────────────────────────
    print(f"\n[1] Loading pipeline: {args.pipeline_ckpt}")
    pipe = Prompt2PromptInpaintPipeline.from_pretrained(
        args.pipeline_ckpt,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    ).to(device)
    pipe.unet.config.addition_embed_type = None
    pipe.set_progress_bar_config(disable=False)
    print(f"  OK — UNet in_channels: {pipe.unet.config.in_channels}")

    # ── [2] Captions ──────────────────────────────────────────────────────────
    print(f"\n[2] Captions...")
    src_cap = args.source_caption
    tgt_cap = args.target_caption

    longclip_embeds = None
    if args.use_long_clip:
        # Thử build Long-CLIP embeddings
        # Nếu thất bại → tự động fallback sang CLIP 77 tokens
        print("  Building Long-CLIP embeddings (248 tokens)...")
        longclip_embeds = build_longclip_embeddings(
            src_cap, tgt_cap, device=device, dtype=dtype)
        if longclip_embeds is None:
            print("  [FALLBACK] Long-CLIP thất bại → dùng CLIP 77 tokens")
            src_cap = safe_truncate(src_cap, pipe.tokenizer)
            tgt_cap = safe_truncate(tgt_cap, pipe.tokenizer)
    else:
        # Standard CLIP — truncate nếu cần
        src_cap = safe_truncate(src_cap, pipe.tokenizer)
        tgt_cap = safe_truncate(tgt_cap, pipe.tokenizer)

    print(f"  Ts: \"{src_cap[:80]}{'...' if len(src_cap)>80 else ''}\"")
    print(f"  Tt: \"{tgt_cap[:80]}{'...' if len(tgt_cap)>80 else ''}\"")

    cross_attn_kwargs = build_cross_attention_kwargs(
        src_cap, tgt_cap, args.p2p_threshold)

    # ── [3] Load Masks ────────────────────────────────────────────────────────
    print(f"\n[3] Loading image + masks...")
    image   = load_and_resize(args.image, args.image_size)
    mf_mask = Image.open(args.mf_mask).convert("L").resize(
        (args.image_size, args.image_size), Image.NEAREST)
    mb_mask = Image.open(args.mb_mask).convert("L").resize(
        (args.image_size, args.image_size), Image.NEAREST)

    mf_np = np.array(mf_mask).astype(np.float32) / 255.0
    mb_np = np.array(mb_mask).astype(np.float32) / 255.0
    print(f"  Mf core:         {(mf_np>0.5).mean()*100:.1f}%")
    print(f"  Mb bbox:         {(mb_np>0.5).mean()*100:.1f}%")
    print(f"  Mb\\Mf transit:   {((mb_np>0.5)&(mf_np<0.5)).mean()*100:.1f}%  "
          f"s={args.soft_mask_value}")

    # Save inputs cho debug
    image.save(os.path.join(args.output_dir,   "input_resized.png"))
    mf_mask.save(os.path.join(args.output_dir, "input_Mf.png"))
    mb_mask.save(os.path.join(args.output_dir, "input_Mb.png"))

    # ── [4] Generate ──────────────────────────────────────────────────────────
    print(f"\n[4] Generating...")
    print(f"  3.1 SDEdit init    (strength={args.strength})")
    print(f"  3.2 P2P cache      (Ts → cross-attn maps)")
    print(f"  3.3 Denoising      ({args.steps} steps, Tt + P2P inject)")
    print(f"  3.4 Alt. blending  Mf=1.0 | Mb\\Mf={args.soft_mask_value} | bg=0.0")
    print(f"  3.5 VAE decode     → target image")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Build call kwargs
    call_kwargs = dict(
        image          = image,
        mask_image     = mf_mask,              # Mf: SAM fine mask
        temp_mask      = mb_mask,              # Mb: BBox mask
        soft_mask      = args.soft_mask_value, # s:  transition value
        num_inference_steps = args.steps,
        strength       = args.strength,
        guidance_scale = args.guidance_scale,
        cross_attention_kwargs = cross_attn_kwargs,
        output_type    = "pil",
    )

    if longclip_embeds is not None:
        # Bypass encode_prompt() — truyền embeddings trực tiếp
        # prompt=None bắt buộc khi dùng prompt_embeds
        call_kwargs["prompt"] = None
        call_kwargs["prompt_embeds"]        = longclip_embeds["prompt_embeds"]
        call_kwargs["pooled_prompt_embeds"] = longclip_embeds["pooled_prompt_embeds"]
        print("  Token mode: Long-CLIP 248 tokens ✓")
    else:
        call_kwargs["prompt"] = [src_cap, tgt_cap]
        print("  Token mode: CLIP 77 tokens")

    out = pipe(**call_kwargs).images
    # out[0] = source reconstruction  (Ts pass)
    # out[1] = target edited image    (Tt + P2P + alternating blend)

    # ── [5] Save ──────────────────────────────────────────────────────────────
    print(f"\n[5] Saving → {args.output_dir}/")
    tag    = "_longclip" if longclip_embeds else "_clip77"
    prefix = (f"s{args.soft_mask_value}_p2p{args.p2p_threshold}"
              f"_str{args.strength}_seed{args.seed}{tag}")

    out[0].save(os.path.join(args.output_dir, f"{prefix}_src_recon.png"))
    out[1].save(os.path.join(args.output_dir, f"{prefix}_target.png"))
    save_comparison(image, mf_mask, mb_mask, out[0], out[1],
                    args.output_dir, prefix)

    print(f"  {prefix}_src_recon.png   ← source reconstruction")
    print(f"  {prefix}_target.png      ← TARGET ảnh đã edit")
    print(f"  {prefix}_comparison.png  ← 5-panel comparison")
    print(f"\n{'='*60}  Done!\n")


if __name__ == "__main__":
    main()