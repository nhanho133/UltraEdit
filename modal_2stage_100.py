"""
modal_2stage_100.py — Two-Stage 100-seed Pipeline
===================================================

Stage 1 (100 seeds): Tìm RECONSTRUCTION tốt nhất
  - Chạy prompt=[src_cap, src_cap] (identity inpainting)
  - Đo PSNR_fg của out[0] vs ảnh gốc TRONG vùng foreground mask
  - Chọn seed cho PSNR_fg cao nhất → best_recon.png

Stage 2 (100 seeds): Edit từ best_recon
  - Input: best_recon.png (thay vì ảnh gốc thô)
  - Chạy prompt=[src_cap, tgt_cap], random hóa hyperparams như GT
  - Filter: clip_img≥0.70, clip_dir≥0.22, dino≥0.40, clip_text≥0.20
  - Output: top-3 by clip_sim_dir (như data_generation.py GT)

Usage:
  modal run modal_2stage_100.py::main \\
    --image-path eval_data/sample_ue/image.png \\
    --mf-mask    eval_data/sample_ue/Mf_mask.png \\
    --mb-mask    eval_data/sample_ue/Mb_mask.png \\
    --source-caption "Two giraffes are next to a tall tree." \\
    --target-caption "Two giraffes are next to a colourful rainbow tree." \\
    --out-dir region_output_2stage
"""

import modal
import os
import io
import sys
from pathlib import Path

app = modal.App("ultraedit-2stage")

# ── Docker Image — identical to modal_sample100.py (reuses cache) ─────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget", "libgl1-mesa-glx", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev",
    )
    .pip_install("numpy<2")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.36.2",
        "accelerate>=0.24.0",
        "huggingface_hub>=0.19.0",
        "safetensors",
    )
    .pip_install(
        "Pillow>=9.0.0",
        "opencv-python-headless",
        "open-clip-torch>=2.20.0",
        "scipy",
        "ftfy",
        "regex",
        "tqdm",
        "einops",
        "scikit-image",
        "openai-clip",
        "matplotlib",
    )
    .run_commands(
        "git clone https://github.com/HaozheZhao/UltraEdit.git /repo/UltraEdit",
        "git clone https://github.com/beichenzbc/Long-CLIP.git /repo/UltraEdit/Long-CLIP",
        "cd /repo/UltraEdit/Long-CLIP && pip install -r requirements.txt 2>/dev/null || true",
    )
        # data_generation: local version has LongCLIP max_num_words + P2P embedding fixes
        .add_local_dir(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation"),
            remote_path="/repo/UltraEdit/data_generation",
        )
        # Override broken pipeline_loading_utils.py from GitHub clone with local fixed version
        # (GitHub clone has empty try: block at line 48 → IndentationError at runtime)
        .add_local_file(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "diffusers/src/diffusers/pipelines/pipeline_loading_utils.py"),
            remote_path="/repo/UltraEdit/diffusers/src/diffusers/pipelines/pipeline_loading_utils.py",
        )
    )

model_cache = modal.Volume.from_name("ultraedit-model-cache", create_if_missing=True)
CACHE_DIR  = "/cache"
HF_CACHE   = "/cache/huggingface"
GPU_CONFIG = "A10G"

# ── Filter thresholds ─────────────────────────────────────────────────────────
# clip_sim_image: CLIP-I(src, edit) — lowered to 0.45 because text-change tasks
#   legitimately reduce CLIP-I (CLIP encodes semantic text content)
CLIP_IMG_THRESH = 0.45
CLIP_DIR_THRESH = 0.22
CLIP_THRESH     = 0.20
DINOV2_THRESH   = 0.40
MAX_OUT_SAMPLES = 3


# ═════════════════════════════════════════════════════════════════════════════
# REMOTE FUNCTION — 2 stages trong 1 container (reuse loaded models)
# ═════════════════════════════════════════════════════════════════════════════
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: model_cache},
    timeout=2400,    # 40 min cho cả 2 stages
    memory=20480,
)
def run_2stage(
    image_bytes:    bytes,
    mf_mask_bytes:  bytes,
    mb_mask_bytes:  bytes,
    source_caption: str,
    target_caption: str,
    n_seeds_s1:     int   = 100,
    n_seeds_s2:     int   = 100,
    pipeline_ckpt:  str   = "stabilityai/sdxl-turbo",
    image_size:     int   = 512,
    min_p2p:        float = 0.1,
    max_p2p:        float = 0.9,
    longclip_encoder: str = "clip77",  # "clip77" | "zer0int" (LongCLIP-GmP, CLIP-G zeroed)
) -> dict:
    import torch
    import random
    import numpy as np
    import warnings
    import logging
    import clip
    import torch.nn.functional as F
    from PIL import Image
    from skimage.metrics import structural_similarity as sk_ssim

    warnings.filterwarnings("ignore")
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    _orig = logging.Logger.warning
    def _filtered(self, msg, *a, **kw):
        if "cross_attention_kwargs" in str(msg) and "not expected" in str(msg):
            return
        _orig(self, msg, *a, **kw)
    logging.Logger.warning = _filtered

    device = "cuda"
    dtype  = torch.float16

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n{'='*60}")
    print(f"  UltraEdit 2-Stage Pipeline")
    print(f"  GPU: {gpu_name}  VRAM: {vram_gb:.1f} GB")
    print(f"  Stage1: {n_seeds_s1} seeds (reconstruction)")
    print(f"  Stage2: {n_seeds_s2} seeds (editing from best_recon)")
    print(f"{'='*60}")

    sys.path.insert(0, "/repo/UltraEdit/diffusers/src")
    sys.path.insert(0, "/repo/UltraEdit/data_generation")
    from sdxl_p2p_pipeline import Prompt2PromptInpaintPipeline

    # ── Helpers ───────────────────────────────────────────────────────────────
    def bytes_to_pil(b, mode="RGB"):
        return Image.open(io.BytesIO(b)).convert(mode)

    def resize_center(img, size):
        w, h = img.size
        scale = size / min(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        w, h = img.size
        return img.crop(((w-size)//2, (h-size)//2, (w+size)//2, (h+size)//2))

    def pil_to_bytes(pil_img, fmt="PNG"):
        buf = io.BytesIO()
        pil_img.save(buf, format=fmt)
        return buf.getvalue()

    def compute_psnr_fg(img_pred, img_gt, mask_pil):
        """PSNR trong vùng foreground mask (mask > 128)"""
        a = np.array(img_pred).astype(np.float64) / 255.0
        b = np.array(img_gt  ).astype(np.float64) / 255.0
        m = (np.array(mask_pil) > 128)
        if m.sum() == 0:
            return 0.0
        diff = (a[m] - b[m]) ** 2
        mse  = diff.mean()
        if mse < 1e-10:
            return 100.0
        return float(20 * np.log10(1.0 / np.sqrt(mse)))

    def compute_psnr_full(img_pred, img_gt):
        a = np.array(img_pred).astype(np.float64) / 255.0
        b = np.array(img_gt  ).astype(np.float64) / 255.0
        mse = ((a - b) ** 2).mean()
        return float(20 * np.log10(1.0 / np.sqrt(max(mse, 1e-10))))

    # ── Load images ────────────────────────────────────────────────────────────
    src_img  = resize_center(bytes_to_pil(image_bytes), image_size)
    mf_mask  = bytes_to_pil(mf_mask_bytes, "L").resize((image_size, image_size), Image.NEAREST)
    mb_mask  = bytes_to_pil(mb_mask_bytes, "L").resize((image_size, image_size), Image.NEAREST)
    src_np   = np.array(src_img).astype(np.float32) / 255.0
    src_t    = torch.from_numpy(src_np).permute(2,0,1)   # [3,H,W] float [0,1]

    # ── Load Diffusion Pipeline ────────────────────────────────────────────────
    print("\n[Load] SDXL-Turbo P2P pipeline...")
    pipe = Prompt2PromptInpaintPipeline.from_pretrained(
        pipeline_ckpt,
        torch_dtype=dtype,
        variant="fp16",
        cache_dir=HF_CACHE,
    ).to(device)
    pipe.unet.config.addition_embed_type = None
    pipe.set_progress_bar_config(disable=True)
    print(f"  UNet in_channels={pipe.unet.config.in_channels} ✓")
    print(f"  longclip_encoder={longclip_encoder}")

    # ── LongCLIP-GmP embeddings (pre-compute once for all S2 seeds) ───────────
    gmp_src_embeds = None   # dict with prompt_embeds + pooled for [src, src]
    gmp_tgt_embeds = None   # dict with prompt_embeds + pooled for [src, tgt]
    MAX_LC_TOKENS  = 248
    CLIP_L_DIM     = 768
    CLIP_G_DIM     = 1280

    if longclip_encoder == "zer0int":
        print("[Load] LongCLIP-GmP ViT-L-14 (zer0int)...")
        from transformers import CLIPTextModel, CLIPTokenizer
        _lc_tok = CLIPTokenizer.from_pretrained(
            "zer0int/LongCLIP-GmP-ViT-L-14", cache_dir=HF_CACHE)
        _lc_enc = CLIPTextModel.from_pretrained(
            "zer0int/LongCLIP-GmP-ViT-L-14", cache_dir=HF_CACHE
        ).to(device).to(dtype).eval()
        print(f"  max_position_embeddings={_lc_enc.config.max_position_embeddings}")

        def _gmp_encode_pair(cap_a, cap_b):
            """Return dict(prompt_embeds [2,248,2048], pooled [2,1280]) for [cap_a, cap_b]."""
            embeds = []
            for cap in [cap_a, cap_b]:
                inp = _lc_tok(
                    cap, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=MAX_LC_TOKENS,
                ).to(device)
                with torch.no_grad():
                    h = _lc_enc(**inp).last_hidden_state.to(dtype)  # [1,248,768]
                g0 = torch.zeros(1, MAX_LC_TOKENS, CLIP_G_DIM, device=device, dtype=dtype)
                embeds.append(torch.cat([h, g0], dim=-1))           # [1,248,2048]
            pe     = torch.cat(embeds, dim=0)                       # [2,248,2048]
            pooled = torch.zeros(2, CLIP_G_DIM, device=device, dtype=dtype)
            return {"prompt_embeds": pe, "pooled_prompt_embeds": pooled}

        # Pre-compute once for both stages
        gmp_src_embeds = _gmp_encode_pair(source_caption, source_caption)  # stage 1 identity
        gmp_tgt_embeds = _gmp_encode_pair(source_caption, target_caption)  # stage 2 edit
        del _lc_enc   # free VRAM after encoding
        torch.cuda.empty_cache()
        print("  LongCLIP-GmP embeddings pre-computed ✓")

    elif longclip_encoder == "chunked":
        # ── Ex3: SeaArt-style chunked CLIP (native SDXL weights) ──────────────
        # Thuật toán giống SeaArtLab/ComfyUI-Long-CLIP:
        #   1. Tokenize caption → raw token ids (không giới hạn 77)
        #   2. Chia thành các chunk 75 tokens (+ BOS + EOS = 77/chunk)
        #   3. Encode CLIP-L và CLIP-G riêng từng chunk
        #   4. CONCATENATE theo token dimension (không average)
        # Output: [2, N*77, 2048] — UNet cross-attn attend tới toàn bộ N*77 tokens
        # Không cần load model mới — dùng lại pipe.text_encoder + pipe.text_encoder_2
        print("[Load] SeaArt chunked CLIP (SDXL native, concatenate chunks)...")

        def _chunked_encode_pair(cap_a, cap_b):
            """SeaArt approach: chunk → encode → CONCAT along token dim.
            Returns dict(prompt_embeds [2, N*77, 2048], pooled [2, 1280]).
            """
            CHUNK = 75  # 75 content tokens + BOS + EOS = 77 per chunk

            def _enc_L(text):
                """CLIP-L: encode each chunk → cat → [1, N*77, 768]"""
                raw = pipe.tokenizer(text, add_special_tokens=False)["input_ids"]
                if not raw:
                    raw = [pipe.tokenizer.unk_token_id]
                chunks = [raw[i:i+CHUNK] for i in range(0, len(raw), CHUNK)]
                hiddens = []
                for chunk in chunks:
                    seq  = ([pipe.tokenizer.bos_token_id] + chunk
                            + [pipe.tokenizer.eos_token_id])
                    seq  = seq[:77]
                    seq += [pipe.tokenizer.pad_token_id] * (77 - len(seq))
                    ids  = torch.tensor([seq], device=device)
                    with torch.no_grad():
                        h = pipe.text_encoder(
                            ids, output_hidden_states=True
                        ).hidden_states[-2].to(dtype)   # [1, 77, 768]
                    hiddens.append(h)
                return torch.cat(hiddens, dim=1)         # [1, N*77, 768]

            def _enc_G(text):
                """CLIP-G: encode each chunk → cat → [1, N*77, 1280]
                           pooled = last chunk EOS hidden state → [1280]"""
                raw = pipe.tokenizer_2(text, add_special_tokens=False)["input_ids"]
                if not raw:
                    raw = [pipe.tokenizer_2.unk_token_id]
                chunks = [raw[i:i+CHUNK] for i in range(0, len(raw), CHUNK)]
                hiddens, pooleds = [], []
                for chunk in chunks:
                    seq     = ([pipe.tokenizer_2.bos_token_id] + chunk
                               + [pipe.tokenizer_2.eos_token_id])
                    seq     = seq[:77]
                    seq    += [pipe.tokenizer_2.pad_token_id] * (77 - len(seq))
                    ids     = torch.tensor([seq], device=device)
                    eos_pos = min(len(chunk) + 1, 76)
                    with torch.no_grad():
                        out = pipe.text_encoder_2(
                            ids, output_hidden_states=True
                        )
                        h = out.hidden_states[-2].to(dtype)              # [1,77,1280]
                        p = out.last_hidden_state[0, eos_pos].to(dtype)  # [1280]
                    hiddens.append(h)
                    pooleds.append(p)
                # pooled = EOS embedding of the LAST chunk (SeaArt convention)
                return (torch.cat(hiddens, dim=1),   # [1, N*77, 1280]
                        pooleds[-1])                 # [1280]

            results = []
            for cap in [cap_a, cap_b]:
                h_L, (h_G, pooled_one) = _enc_L(cap), _enc_G(cap)
                # align token lengths (CLIP-L vs CLIP-G may differ by 1 chunk)
                n = min(h_L.shape[1], h_G.shape[1])
                combined = torch.cat([h_L[:, :n, :], h_G[:, :n, :]], dim=-1)  # [1, N*77, 2048]
                results.append((combined, pooled_one))

            pe     = torch.cat([r[0] for r in results], dim=0)    # [2, N*77, 2048]
            pooled = torch.stack([r[1] for r in results], dim=0)  # [2, 1280]
            return {"prompt_embeds": pe, "pooled_prompt_embeds": pooled,
                    "n_tokens": pe.shape[1]}  # store for max_num_words

        gmp_src_embeds = _chunked_encode_pair(source_caption, source_caption)
        gmp_tgt_embeds = _chunked_encode_pair(source_caption, target_caption)
        n_tok_src = gmp_src_embeds.pop("n_tokens")
        n_tok_tgt = gmp_tgt_embeds.pop("n_tokens")
        CHUNKED_N_TOKENS = max(n_tok_src, n_tok_tgt)  # for max_num_words
        print(f"  SeaArt chunked embeddings: {n_tok_src} tokens (src/src), "
              f"{n_tok_tgt} tokens (src/tgt)  ✓")
    else:
        CHUNKED_N_TOKENS = 77  # unused for non-chunked encoders

    # ── Load CLIP ViT-L/14 ─────────────────────────────────────────────────────
    print("[Load] CLIP ViT-L/14...")
    clip_model, _ = clip.load("ViT-L/14", device=device, download_root=HF_CACHE)
    clip_model.eval().requires_grad_(False)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(3,1,1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(3,1,1)

    def clip_encode_img(img_t):
        x = img_t.unsqueeze(0).to(device)
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
        x = (x - clip_mean) / clip_std
        with torch.no_grad():
            feat = clip_model.encode_image(x.half())
        return (feat / feat.norm(dim=-1, keepdim=True)).float()

    def clip_encode_txt(text):
        toks = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            feat = clip_model.encode_text(toks)
        return (feat / feat.norm(dim=-1, keepdim=True)).float()

    # ── Load DINOv2 ──────────────────────────────────────────────────────────
    print("[Load] DINOv2 ViT-L/14-reg...")
    dinov2 = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14_reg",
        source="github",
    ).to(device).eval().requires_grad_(False)

    def dinov2_encode(img_t):
        x = img_t.unsqueeze(0).to(device)
        x = F.interpolate(x, size=(518, 518), mode="bicubic", align_corners=False)
        with torch.no_grad():
            feat = dinov2(x.float())
        return (feat / feat.norm(dim=-1, keepdim=True)).float()

    # Pre-compute original source features (for Stage 2 CLIP metrics)
    print("[Load] Pre-computing source features...")
    src_clip_feat = clip_encode_img(src_t)
    src_dino_feat = dinov2_encode(src_t)
    txt0_feat     = clip_encode_txt(source_caption)
    txt1_feat     = clip_encode_txt(target_caption)
    print(f"  CLIP: {src_clip_feat.shape}  DINOv2: {src_dino_feat.shape}")

    # ── Diff words for P2P replace ─────────────────────────────────────────────
    src_words = source_caption.split()
    tgt_words = target_caption.split()
    diff = [b for a, b in zip(src_words, tgt_words) if a != b]
    edit_type_s2 = "replace" if (diff and len(src_words) == len(tgt_words)) else "refine"

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: Find best source reconstruction (prompt=[src, src])
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  STAGE 1: {n_seeds_s1} seeds — best RECONSTRUCTION (PSNR_fg)")
    print(f"{'─'*60}")

    s1_results = []
    best_psnr        = -1.0
    best_recon_bytes = None   # image bytes for the best src_recon
    best_recon_seed  = None
    best_recon_idx   = -1
    best_recon_img   = src_img  # fallback = original

    for i in range(n_seeds_s1):
        seed = torch.randint(1 << 32, ()).item()
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        random.seed(seed); np.random.seed(seed)

        p2p_thr  = min_p2p + random.random() * (max_p2p - min_p2p)
        steps    = random.choice([10, 14])
        cfg      = random.choice([0.0, 0.2, 0.4, 0.6])
        soft     = random.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.8])
        use_soft = random.choice([True, False])

        # Identity: both prompts = source caption
        cross_attn = {
            "edit_type":       "refine",
            "n_self_replace":  p2p_thr,
            "n_cross_replace": p2p_thr,
            "prompts":         [source_caption, source_caption],
            "max_num_words":   (248 if longclip_encoder == "zer0int"
                                 else CHUNKED_N_TOKENS if longclip_encoder == "chunked"
                                 else 77),
        }
        call_kw = dict(
            prompt               = [source_caption, source_caption],
            image                = src_img,
            mask_image           = mf_mask,
            temp_mask            = mb_mask,
            num_inference_steps  = steps,
            guidance_scale       = cfg,
            cross_attention_kwargs = cross_attn,
            output_type          = "pil",
        )
        if use_soft:
            call_kw["soft_mask"]   = soft
            call_kw["mask_choice"] = "wo_final_layer"

        # Inject LongCLIP-GmP embeddings for stage 1 if requested
        if gmp_src_embeds is not None:
            call_kw["prompt"]                 = None
            call_kw["prompt_embeds"]          = gmp_src_embeds["prompt_embeds"]
            call_kw["pooled_prompt_embeds"]   = gmp_src_embeds["pooled_prompt_embeds"]

        try:
            out        = pipe(**call_kw).images
            recon_img  = out[0]   # index 0 = src reconstruction
        except Exception as e:
            s1_results.append({"seed": seed, "idx": i, "psnr_fg": -1.0, "error": str(e)})
            continue

        psnr_fg   = compute_psnr_fg(recon_img, src_img, mf_mask)
        psnr_full = compute_psnr_full(recon_img, src_img)

        recon_np  = np.array(recon_img).astype(np.float32) / 255.0
        ssim_c = [sk_ssim(src_np[:,:,c], recon_np[:,:,c], data_range=1.0) for c in range(3)]
        ssim_val  = float(np.mean(ssim_c))

        rec = dict(
            seed=seed, idx=i, steps=steps, cfg=cfg,
            soft=soft if use_soft else None, use_soft=use_soft, p2p_thr=p2p_thr,
            psnr_fg=psnr_fg, psnr_full=psnr_full, ssim=ssim_val,
        )
        s1_results.append(rec)

        if psnr_fg > best_psnr:
            best_psnr        = psnr_fg
            best_recon_bytes = pil_to_bytes(recon_img)
            best_recon_idx   = i
            best_recon_seed  = seed
            best_recon_img   = recon_img   # PIL — keep for stage 2

        if (i+1) % 10 == 0:
            print(f"  [{i+1:3d}/{n_seeds_s1}]  best_psnr_fg={best_psnr:.2f} dB  "
                  f"(seed #{best_recon_idx+1})")

    print(f"\n  Stage 1 done: best PSNR_fg = {best_psnr:.2f} dB  "
          f"(seed {best_recon_seed}, idx #{best_recon_idx+1})")
    if best_recon_bytes is None:
        print("  [WARN] No valid reconstruction — using original image for Stage 2")
        best_recon_bytes = pil_to_bytes(src_img)
        best_recon_img   = src_img

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: Edit từ best_recon  (prompt=[src, tgt])
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  STAGE 2: {n_seeds_s2} seeds — EDITING from best_recon")
    print(f"  Thresholds: clip_img≥{CLIP_IMG_THRESH} dir≥{CLIP_DIR_THRESH} dino≥{DINOV2_THRESH}")
    print(f"{'─'*60}")

    # Pre-compute best_recon features (for CLIP image-image metric in stage 2)
    recon_t         = torch.from_numpy(np.array(best_recon_img).astype(np.float32)/255.0).permute(2,0,1)
    recon_clip_feat = clip_encode_img(recon_t)
    recon_dino_feat = dinov2_encode(recon_t)

    s2_results = []

    for i in range(n_seeds_s2):
        seed = torch.randint(1 << 32, ()).item()
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        random.seed(seed); np.random.seed(seed)

        p2p_thr  = min_p2p + random.random() * (max_p2p - min_p2p)
        steps    = random.choice([10, 14])
        cfg      = random.choice([0.0, 0.2, 0.4, 0.6])
        soft     = random.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.8])
        use_soft = random.choice([True, False])
        strength = random.choice([0.5, 0.7, 0.85, 0.99])  # higher = more creative edit

        n_cross = ({",".join(diff[:5]): p2p_thr, "default_": 1.0}
                   if edit_type_s2 == "replace" and diff else p2p_thr)
        cross_attn = {
            "edit_type":       edit_type_s2,
            "n_self_replace":  p2p_thr,
            "n_cross_replace": n_cross,
            "prompts":         [source_caption, target_caption],
            "max_num_words":   (248 if longclip_encoder == "zer0int"
                                 else CHUNKED_N_TOKENS if longclip_encoder == "chunked"
                                 else 77),
        }
        call_kw = dict(
            prompt               = [source_caption, target_caption],
            image                = best_recon_img,   # ← stage 2 input = best_recon
            mask_image           = mf_mask,
            temp_mask            = mb_mask,
            num_inference_steps  = steps,
            guidance_scale       = cfg,
            strength             = strength,
            cross_attention_kwargs = cross_attn,
            output_type          = "pil",
        )
        if use_soft:
            call_kw["soft_mask"]   = soft
            call_kw["mask_choice"] = "wo_final_layer"

        # Inject LongCLIP-GmP embeddings for stage 2 if requested
        if gmp_tgt_embeds is not None:
            call_kw["prompt"]                 = None
            call_kw["prompt_embeds"]          = gmp_tgt_embeds["prompt_embeds"]
            call_kw["pooled_prompt_embeds"]   = gmp_tgt_embeds["pooled_prompt_embeds"]

        try:
            out        = pipe(**call_kw).images
            edited_img = out[1]
        except Exception as e:
            s2_results.append({"seed": seed, "idx": i, "clip_sim_dir": -1.0, "error": str(e)})
            continue

        edit_t    = torch.from_numpy(np.array(edited_img).astype(np.float32)/255.0).permute(2,0,1)
        edit_clip = clip_encode_img(edit_t)
        edit_dino = dinov2_encode(edit_t)

        # Metrics vs ORIGINAL source (not recon) — consistent with GT filtering
        clip_sim_0     = F.cosine_similarity(src_clip_feat,  txt0_feat).item()
        clip_sim_1     = F.cosine_similarity(edit_clip,      txt1_feat).item()
        clip_sim_image = F.cosine_similarity(src_clip_feat,  edit_clip).item()
        clip_sim_dir   = F.cosine_similarity(
            edit_clip - src_clip_feat,
            txt1_feat - txt0_feat
        ).item()
        dinov2_sim     = F.cosine_similarity(src_dino_feat, edit_dino).item()

        # Also compute vs recon (for reference)
        clip_dir_recon = F.cosine_similarity(
            edit_clip - recon_clip_feat,
            txt1_feat - txt0_feat
        ).item()
        dino_recon     = F.cosine_similarity(recon_dino_feat, edit_dino).item()

        edit_np  = np.array(edited_img).astype(np.float32) / 255.0
        ssim_val = float(np.mean([sk_ssim(src_np[:,:,c], edit_np[:,:,c], data_range=1.0) for c in range(3)]))

        rec = dict(
            seed=seed, idx=i, steps=steps, cfg=cfg,
            soft=soft if use_soft else None, use_soft=use_soft, p2p_thr=p2p_thr,
            strength=strength,
            clip_sim_0=clip_sim_0, clip_sim_1=clip_sim_1,
            clip_sim_image=clip_sim_image, clip_sim_dir=clip_sim_dir,
            clip_dir_recon=clip_dir_recon,
            dinov2_sim=dinov2_sim, dino_recon=dino_recon,
            ssim=ssim_val,
            pass_filter=(
                clip_sim_image >= CLIP_IMG_THRESH and
                clip_sim_dir   >= CLIP_DIR_THRESH and
                clip_sim_0     >= CLIP_THRESH     and
                clip_sim_1     >= CLIP_THRESH     and
                dinov2_sim     >= DINOV2_THRESH
            ),
        )
        # Always store image bytes — top-3 saved regardless of filter
        rec["image_bytes"]     = pil_to_bytes(edited_img)
        rec["src_recon_bytes"] = pil_to_bytes(out[0])

        s2_results.append(rec)

        if (i+1) % 10 == 0:
            n_pass = sum(r.get("pass_filter", False) for r in s2_results)
            best_d = max((r.get("clip_sim_dir", -1) for r in s2_results), default=0)
            print(f"  [{i+1:3d}/{n_seeds_s2}]  pass={n_pass}  best_dir={best_d:.3f}")

    survivors = [r for r in s2_results if r.get("pass_filter", False)]
    # Best-effort top-3: passed filter first, else top-3 by clip_dir from all valid seeds
    all_valid = [r for r in s2_results if "clip_sim_dir" in r and "error" not in r and "image_bytes" in r]
    all_valid.sort(key=lambda r: r["clip_sim_dir"], reverse=True)
    top = survivors[:MAX_OUT_SAMPLES] if survivors else all_valid[:MAX_OUT_SAMPLES]

    print(f"\n  Stage 2 done: {len(survivors)}/{n_seeds_s2} passed strict filter")
    print(f"  Saving top-{len(top)} by clip_dir ({'PASSED' if survivors else 'BEST-EFFORT, no strict pass'})")
    for j, r in enumerate(top):
        pflag = "✓" if r.get("pass_filter") else "~"
        print(f"  {pflag}#{j+1}  seed={r['seed']}  dir={r['clip_sim_dir']:.3f}  "
              f"img={r['clip_sim_image']:.3f}  dino={r['dinov2_sim']:.3f}  "
              f"steps={r['steps']}  cfg={r['cfg']}  soft={r['soft']}")

    return {
        "stage1_results":    s1_results,
        "stage2_results":    s2_results,
        "top":               top,
        "best_recon_bytes":  best_recon_bytes,
        "best_recon_seed":   best_recon_seed,
        "best_psnr_fg":      best_psnr,
        "n_pass_s2":         len(survivors),
    }


# ═════════════════════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════
@app.local_entrypoint()
def main(
    image_path:     str   = "eval_data/sample_ue/image.png",
    mf_mask:        str   = "eval_data/sample_ue/Mf_mask.png",
    mb_mask:        str   = "eval_data/sample_ue/Mb_mask.png",
    source_caption: str   = "Two giraffes are next to a tall tree.",
    target_caption: str   = "Two giraffes are next to a colourful rainbow tree.",
    n_seeds_s1:     int   = 100,
    n_seeds_s2:     int   = 100,
    out_dir:        str   = "region_output_2stage",
    pipeline_ckpt:  str   = "stabilityai/sdxl-turbo",
    longclip_encoder: str = "clip77",  # "clip77" | "zer0int" | "both"
):
    import json
    import numpy as np
    from PIL import Image, ImageDraw

    # ── Determine which encoders to run ───────────────────────────────────────
    if longclip_encoder == "both":
        encoders = [("clip77", "1_clip77"), ("zer0int", "2_longclip248")]
    else:
        encoders = [(longclip_encoder, longclip_encoder)]

    image_bytes   = Path(image_path).read_bytes()
    mf_bytes      = Path(mf_mask).read_bytes()
    mb_bytes      = Path(mb_mask).read_bytes()

    print(f"\n{'='*60}")
    print(f"  UltraEdit 2-Stage Pipeline")
    print(f"  Image  : {image_path}")
    print(f"  Source : {source_caption}")
    print(f"  Target : {target_caption}")
    print(f"  Encoders: {[e[0] for e in encoders]}")
    print(f"  Stage1 : {n_seeds_s1} seeds | Stage2 : {n_seeds_s2} seeds")
    print(f"{'='*60}")

    # ── Spawn all jobs in parallel using .spawn() (non-blocking) ────────────
    futures = []
    for enc, tag in encoders:
        print(f"  Launching job: encoder={enc}  tag={tag}")
        fut = run_2stage.spawn(
            image_bytes      = image_bytes,
            mf_mask_bytes    = mf_bytes,
            mb_mask_bytes    = mb_bytes,
            source_caption   = source_caption,
            target_caption   = target_caption,
            n_seeds_s1       = n_seeds_s1,
            n_seeds_s2       = n_seeds_s2,
            pipeline_ckpt    = pipeline_ckpt,
            longclip_encoder = enc,
        )
        futures.append((enc, tag, fut))

    # ── Collect results and save ──────────────────────────────────────────────
    SIZE = 512
    src_img = Image.open(image_path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    mf_v    = Image.open(mf_mask).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    gt_path = Path(image_path).parent / "gt_edited.png"
    gt_img  = Image.open(gt_path).convert("RGB").resize((SIZE, SIZE)) if gt_path.exists() else None

    for enc, tag, fut in futures:
        print(f"\n{'='*60}")
        print(f"  Collecting result: encoder={enc}  tag={tag}")
        result = fut.get()  # wait for spawned job to complete

        save_dir = Path(out_dir) / tag
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "stage1").mkdir(exist_ok=True)
        (save_dir / "stage2").mkdir(exist_ok=True)

        s1_res = result["stage1_results"]
        s2_res = result["stage2_results"]
        top    = result["top"]
        n_pass = result["n_pass_s2"]

        # ── Stage 1 ───────────────────────────────────────────────────────────
        best_recon = Image.open(io.BytesIO(result["best_recon_bytes"])).convert("RGB")
        best_recon.save(save_dir / "stage1" / f"best_recon_psnr{result['best_psnr_fg']:.2f}.png")

        panels_s1 = [(src_img, "Original"), (best_recon, f"Best Recon\nPSNR_fg={result['best_psnr_fg']:.2f} dB")]
        if gt_img: panels_s1.insert(2, (gt_img, "Dataset GT"))
        comp1 = Image.new("RGB", (SIZE * len(panels_s1), SIZE), (20, 20, 20))
        draw1 = ImageDraw.Draw(comp1)
        for i, (im, lbl) in enumerate(panels_s1):
            comp1.paste(im.resize((SIZE, SIZE)), (i * SIZE, 0))
            first = lbl.split("\n")[0]
            draw1.rectangle([(i*SIZE+2, 2), (i*SIZE + len(first)*8+8, 22)], fill=(0,0,0))
            draw1.text((i*SIZE+5, 4), first, fill="white")
            if "\n" in lbl: draw1.text((i*SIZE+5, 20), lbl.split("\n")[1], fill=(200,200,200))
        comp1.save(save_dir / "stage1" / "comparison_recon.png")

        psnrs = [r["psnr_fg"] for r in s1_res if r.get("psnr_fg", -1) >= 0]
        if psnrs:
            print(f"[{tag}] Stage1 PSNR_fg: mean={np.mean(psnrs):.2f}  max={np.max(psnrs):.2f}  min={np.min(psnrs):.2f}")
        else:
            print(f"[{tag}] Stage1: all seeds errored (0 valid reconstructions)")
        print(f"[{tag}] Best seed: {result['best_recon_seed']}  PSNR_fg={result['best_psnr_fg']:.2f} dB")

        # ── Stage 2 ───────────────────────────────────────────────────────────
        print(f"[{tag}] Stage2: {n_pass}/{n_seeds_s2} passed filter → top {len(top)} saved")
        for j, r in enumerate(top):
            tgt = Image.open(io.BytesIO(r["image_bytes"])).convert("RGB")
            pflag = "pass" if r.get("pass_filter") else "bestef"
            tag2 = f"top{j+1}_{pflag}_seed{r['seed']}_dir{r['clip_sim_dir']:.3f}_steps{r['steps']}_cfg{r['cfg']}"
            tgt.save(save_dir / "stage2" / f"{tag2}.png")

            panels = [
                (src_img,     "Original"),
                (best_recon,  f"Stage1 Recon\nPSNR={result['best_psnr_fg']:.1f}dB"),
                (mf_v.convert("RGB"), "Mask"),
            ]
            if gt_img: panels.append((gt_img, "Dataset GT"))
            panels.append((tgt, f"#{j+1} dir={r['clip_sim_dir']:.3f}\nsteps={r['steps']} cfg={r['cfg']} soft={r['soft']}"))
            W = SIZE * len(panels)
            comp = Image.new("RGB", (W, SIZE), (20, 20, 20))
            draw = ImageDraw.Draw(comp)
            for i, (im, lbl) in enumerate(panels):
                comp.paste(im.resize((SIZE, SIZE)), (i * SIZE, 0))
                first = lbl.split("\n")[0]
                draw.rectangle([(i*SIZE+2, 2), (i*SIZE + len(first)*8+8, 22)], fill=(0,0,0))
                draw.text((i*SIZE+5, 4), first, fill="white")
                if "\n" in lbl: draw.text((i*SIZE+5, 20), lbl.split("\n")[1], fill=(200,200,200))
            comp.save(save_dir / "stage2" / f"comparison_top{j+1}.png")
            print(f"  [{tag}] #{j+1}: clip_dir={r['clip_sim_dir']:.3f}  clip_img={r['clip_sim_image']:.3f}  "
                  f"dino={r['dinov2_sim']:.3f}  steps={r['steps']}  cfg={r['cfg']}")

        # ── Metric distributions ──────────────────────────────────────────────
        s2_valid = [r for r in s2_res if "clip_sim_dir" in r and "error" not in r]
        dirs  = [r["clip_sim_dir"]   for r in s2_valid]
        imgs  = [r["clip_sim_image"] for r in s2_valid]
        dinos = [r["dinov2_sim"]     for r in s2_valid]
        if dirs:
            print(f"[{tag}] clip_sim_dir : mean={np.mean(dirs):.3f}  max={np.max(dirs):.3f}  ≥{CLIP_DIR_THRESH}: {sum(d>=CLIP_DIR_THRESH for d in dirs)}")
            print(f"[{tag}] clip_sim_image: mean={np.mean(imgs):.3f}  max={np.max(imgs):.3f}")
            print(f"[{tag}] dinov2_sim   : mean={np.mean(dinos):.3f}  max={np.max(dinos):.3f}")
        else:
            print(f"[{tag}] No valid Stage 2 results (all seeds errored)")

        # ── Save metrics JSON ─────────────────────────────────────────────────
        clean_s2 = [{k:v for k,v in r.items() if k not in ("image_bytes","src_recon_bytes")} for r in s2_res]
        with open(save_dir / "all_metrics.json", "w") as f:
            json.dump({
                "encoder": enc, "stage1": s1_res, "stage2": clean_s2,
                "best_recon_seed": result["best_recon_seed"],
                "best_psnr_fg": result["best_psnr_fg"], "n_pass_s2": n_pass,
            }, f, indent=2)

        # ── Scatter plot ──────────────────────────────────────────────────────
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            ax = axes[0]
            ax.hist([r["psnr_fg"] for r in s1_res if r.get("psnr_fg",-1)>=0],
                    bins=20, color="steelblue", alpha=0.7, edgecolor="white")
            ax.axvline(result["best_psnr_fg"], color="gold", lw=2, label=f"best={result['best_psnr_fg']:.2f} dB")
            ax.set_xlabel("PSNR_fg (dB)"); ax.set_ylabel("count")
            ax.set_title(f"Stage1 PSNR_fg [{enc}]"); ax.legend()

            ax2 = axes[1]
            fail_r = [r for r in s2_valid if not r.get("pass_filter")]
            pass_r = [r for r in s2_valid if r.get("pass_filter")]
            if fail_r: ax2.scatter([r["clip_sim_dir"] for r in fail_r],[r["dinov2_sim"] for r in fail_r],c="lightcoral",alpha=0.5,s=20,label="fail")
            if pass_r: ax2.scatter([r["clip_sim_dir"] for r in pass_r],[r["dinov2_sim"] for r in pass_r],c="steelblue",alpha=0.8,s=40,label="pass")
            for j,r in enumerate(top): ax2.scatter(r["clip_sim_dir"],r["dinov2_sim"],c="gold",s=150,marker="*",zorder=5,label=f"top-{j+1}" if j==0 else "")
            ax2.axvline(CLIP_DIR_THRESH,color="gray",ls="--",lw=0.8); ax2.axhline(DINOV2_THRESH,color="gray",ls="--",lw=0.8)
            ax2.set_xlabel("clip_sim_dir"); ax2.set_ylabel("dinov2_sim")
            ax2.set_title(f"Stage2 CLIP dir vs DINOv2 [{enc}]\n{n_pass}/{n_seeds_s2} pass"); ax2.legend()

            ax3 = axes[2]
            if fail_r: ax3.scatter([r["clip_sim_image"] for r in fail_r],[r["clip_sim_dir"] for r in fail_r],c="lightcoral",alpha=0.5,s=20,label="fail")
            if pass_r: ax3.scatter([r["clip_sim_image"] for r in pass_r],[r["clip_sim_dir"] for r in pass_r],c="steelblue",alpha=0.8,s=40,label="pass")
            for j,r in enumerate(top): ax3.scatter(r["clip_sim_image"],r["clip_sim_dir"],c="gold",s=150,marker="*",zorder=5)
            ax3.axvline(CLIP_IMG_THRESH,color="gray",ls="--",lw=0.8); ax3.axhline(CLIP_DIR_THRESH,color="gray",ls="--",lw=0.8)
            ax3.set_xlabel("clip_sim_image"); ax3.set_ylabel("clip_sim_dir")
            ax3.set_title(f"Stage2 CLIP img vs dir [{enc}]")

            plt.suptitle(f"2-Stage 100-seed [{enc}] | PSNR_fg={result['best_psnr_fg']:.2f} dB | pass={n_pass}/{n_seeds_s2}", fontsize=10)
            plt.tight_layout()
            plt.savefig(save_dir / "scatter.png", dpi=150)
            print(f"[{tag}] scatter.png saved")
        except Exception as e:
            print(f"  [scatter failed: {e}]")

        print(f"[{tag}] Saved → {save_dir}/")

    print(f"\n{'='*60}  Done!\n")


# ═════════════════════════════════════════════════════════════════════════════
# BATCH ENTRYPOINT — tất cả samples trong eval_data/sample_0*/
# ═════════════════════════════════════════════════════════════════════════════
@app.local_entrypoint()
def run_all(
    longclip_encoder: str = "both",   # "clip77" | "zer0int" | "both"
    n_seeds_s1:       int = 100,
    n_seeds_s2:       int = 100,
    out_dir:          str = "eval_results_2stage",
    samples:          str = "",        # comma-sep subset, e.g. "sample_01,sample_03"
):
    """
    Chạy toàn bộ eval trên tất cả samples:
      modal run modal_2stage_100.py::run_all
      modal run modal_2stage_100.py::run_all --longclip-encoder both
      modal run modal_2stage_100.py::run_all --samples "sample_01,sample_02"
    """
    import json
    import io
    from PIL import Image

    EVAL_DATA = Path("eval_data")

    # Xác định encoder tags
    if longclip_encoder == "all":
        encoders = [("clip77", "1_clip77"), ("zer0int", "2_longclip248"), ("chunked", "3_chunked")]
    elif longclip_encoder == "both":
        encoders = [("clip77", "1_clip77"), ("zer0int", "2_longclip248")]
    elif longclip_encoder == "clip77":
        encoders = [("clip77", "1_clip77")]
    elif longclip_encoder == "zer0int":
        encoders = [("zer0int", "2_longclip248")]
    elif longclip_encoder == "chunked":
        encoders = [("chunked", "3_chunked")]
    else:
        raise ValueError(f"Unknown longclip_encoder: {longclip_encoder!r}. "
                         f"Choose: clip77 | zer0int | chunked | both | all")

    # Load sample dirs (sample_01..05 + sample_ue, etc.)
    sample_dirs = sorted(EVAL_DATA.glob("sample_*/"))
    if samples:
        wanted = {s.strip() for s in samples.split(",")}
        sample_dirs = [s for s in sample_dirs if s.name in wanted]

    print(f"\n{'='*60}")
    print(f"  UltraEdit Batch Eval — {len(sample_dirs)} samples × {len(encoders)} encoders")
    print(f"  Seeds: S1={n_seeds_s1}  S2={n_seeds_s2}")
    print(f"  Out  : {out_dir}/")
    print(f"{'='*60}\n")

    import numpy as np
    SIZE = 512

    # ── Đọc bytes + metadata cho từng sample ─────────────────────────────────
    loaded = []
    for sd in sample_dirs:
        with open(sd / "metadata.json") as f:
            meta = json.load(f)
        loaded.append({
            "sid":         sd.name,
            "meta":        meta,
            "image_bytes": (sd / "image.png").read_bytes(),
            "mf_bytes":    (sd / "Mf_mask.png").read_bytes(),
            "mb_bytes":    (sd / "Mb_mask.png").read_bytes(),
        })
        print(f"  Loaded {sd.name}: {meta.get('source_caption_long','?')[:60]}")

    # ── Outer=encoder, Inner=sample — ex1 hết 5 samples rồi mới sang ex2 ─────
    for enc, tag in encoders:
        print(f"\n{'─'*60}")
        print(f"  Encoder: {enc}  →  {tag}")
        print(f"{'─'*60}")

        for s in loaded:
            sid  = s["sid"]
            meta = s["meta"]
            cap_src = meta["source_caption_long"]
            cap_tgt = meta["target_caption_long"]

            print(f"\n  Running [{tag}] [{sid}]  ...")
            try:
                result = run_2stage.remote(
                    image_bytes      = s["image_bytes"],
                    mf_mask_bytes    = s["mf_bytes"],
                    mb_mask_bytes    = s["mb_bytes"],
                    source_caption   = cap_src,
                    target_caption   = cap_tgt,
                    n_seeds_s1       = n_seeds_s1,
                    n_seeds_s2       = n_seeds_s2,
                    longclip_encoder = enc,
                )
            except Exception as e:
                print(f"  [ERROR] {tag}/{sid}: {e}")
                continue

            save_dir = Path(out_dir) / tag / sid
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "stage1").mkdir(exist_ok=True)
            (save_dir / "stage2").mkdir(exist_ok=True)

            # best_recon
            best_recon = Image.open(io.BytesIO(result["best_recon_bytes"])).convert("RGB")
            best_recon.save(save_dir / "stage1" / f"best_recon_psnr{result['best_psnr_fg']:.2f}.png")

            # top edited images
            top    = result["top"]
            n_pass = result["n_pass_s2"]
            for j, r in enumerate(top):
                tgt   = Image.open(io.BytesIO(r["image_bytes"])).convert("RGB")
                pflag = "pass" if r.get("pass_filter") else "bestef"
                fname = f"top{j+1}_{pflag}_dir{r['clip_sim_dir']:.3f}_steps{r['steps']}_cfg{r['cfg']}.png"
                tgt.save(save_dir / "stage2" / fname)

            # metrics JSON
            clean_s2 = [{k: v for k, v in r.items() if k not in ("image_bytes","src_recon_bytes")}
                        for r in result["stage2_results"]]
            with open(save_dir / "metrics.json", "w") as f:
                json.dump({
                    "encoder":         enc,
                    "best_recon_seed": result["best_recon_seed"],
                    "best_psnr_fg":    result["best_psnr_fg"],
                    "n_pass_s2":       n_pass,
                    "stage1":          result["stage1_results"],
                    "stage2":          clean_s2,
                }, f, indent=2)

            s2_valid = [r for r in result["stage2_results"] if "clip_sim_dir" in r and "error" not in r]
            dirs = [r["clip_sim_dir"] for r in s2_valid]
            if dirs:
                print(f"  [{tag}][{sid}] PSNR_fg={result['best_psnr_fg']:.2f}dB  "
                      f"pass={n_pass}/{n_seeds_s2}  best_dir={max(dirs):.3f}")
            else:
                print(f"  [{tag}][{sid}] PSNR_fg={result['best_psnr_fg']:.2f}dB  "
                      f"pass={n_pass}/{n_seeds_s2}  no valid seeds")
            print(f"  → Saved to {save_dir}/")

    print(f"\n{'='*60}  Done!\n")
