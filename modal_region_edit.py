"""
Modal Deployment — UltraEdit Region-based Edit
===============================================
Chạy region_gen_p2p.py trên Modal với GPU A10G/A100.

Model caching strategy:
  - SDXL-Turbo      → /cache/huggingface/ (HF cache, tự động)
  - Long-CLIP ViT-L → /cache/huggingface/ (HF cache, tự động)
  - OpenCLIP bigG   → /cache/openclip_vitbigG14.pt (manual cache)
  Tất cả đều trong Modal Volume "ultraedit-model-cache"
  → Lần đầu download, lần sau load từ Volume (~10-30s thay vì 5-10 phút)

Setup:
  pip install modal
  modal token new

  # Lần đầu: download tất cả models vào Volume
  modal run modal_region_edit.py::download_models

  # Các lần sau: chạy inference (không download nữa)
  modal run modal_region_edit.py --image-path ... --source-caption ... --target-caption ...
"""

import modal
import os
import io
import sys
from pathlib import Path

# ── Modal App ─────────────────────────────────────────────────────────────────
app = modal.App("ultraedit-region-edit")

# ── Docker Image ──────────────────────────────────────────────────────────────
# Build image với tất cả dependencies của UltraEdit
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget", "libgl1-mesa-glx", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev",
    )
    # NumPy < 2 trước — tránh conflict với torch
    .pip_install("numpy<2")
    .pip_install(
        # PyTorch >= 2.4 (transformers yêu cầu)
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # transformers version cũ — compatible với diffusers fork UltraEdit
        # FLAX_WEIGHTS_NAME bị remove ở transformers>=4.38
        "transformers==4.36.2",
        "accelerate>=0.24.0",
        "huggingface_hub>=0.19.0",
        "safetensors",
    )
    .pip_install(
        # Image + misc
        "Pillow>=9.0.0",
        "opencv-python-headless",
        "open-clip-torch>=2.20.0",
        "scipy",
        "ftfy",
        "regex",
        "tqdm",
        "einops",
    )
    # Clone UltraEdit repo
    .run_commands(
        "git clone https://github.com/HaozheZhao/UltraEdit.git /repo/UltraEdit",
        "git clone https://github.com/beichenzbc/Long-CLIP.git /repo/UltraEdit/Long-CLIP",
        # Install Long-CLIP
        "cd /repo/UltraEdit/Long-CLIP && pip install -r requirements.txt 2>/dev/null || true",
        # Install diffusers fork từ repo (không dùng pip install diffusers)
        "cd /repo/UltraEdit/diffusers && pip install -e . --no-deps 2>/dev/null || true",
    )
    # Overlay local patched data_generation/ over the GitHub clone
    # Includes fix: p2p_prompts fallback from cross_attention_kwargs["prompts"]
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation"),
        remote_path="/repo/UltraEdit/data_generation",
    )
)

# ── Volume để cache models ─────────────────────────────────────────────────────
# Models được cache vào volume → lần sau không download lại
model_cache = modal.Volume.from_name(
    "ultraedit-model-cache", create_if_missing=True)

# ── GPU Config ────────────────────────────────────────────────────────────────
GPU_CONFIG = "A10G"   # Đổi thành "A100" nếu cần memory nhiều hơn

# Paths trong Volume
CACHE_DIR      = "/cache"
HF_CACHE       = "/cache/huggingface"
OPENCLIP_CACHE = "/cache/openclip_vitbigG14.pt"
LONGCLIP_CACHE = "/cache/huggingface/hub/models--BeichenZhang--LongCLIP-L"


# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOAD MODELS  (chạy 1 lần duy nhất để populate Volume)
# modal run modal_region_edit.py::download_models
# ═════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={CACHE_DIR: model_cache},
    timeout=3600,   # 1 giờ cho lần download đầu
    cpu=4,
    memory=16384,
)
def download_models():
    """
    Download tất cả models vào Modal Volume.
    Chỉ cần chạy 1 lần: modal run modal_region_edit.py::download_models
    Sau đó inference sẽ load từ Volume (~10-30s, không download lại).
    """
    import torch
    from huggingface_hub import snapshot_download, hf_hub_download
    sys.path.insert(0, "/repo/UltraEdit/Long-CLIP")
    from open_clip_long import factory as open_clip

    os.makedirs(HF_CACHE, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. SDXL-Turbo
    print("[1/3] Downloading SDXL-Turbo...")
    snapshot_download(
        "stabilityai/sdxl-turbo",
        cache_dir=HF_CACHE,
        ignore_patterns=["*.msgpack", "*.h5", "flax_*"],
    )
    print("  SDXL-Turbo OK")

    # 2. Long-CLIP ViT-L
    print("[2/3] Downloading Long-CLIP ViT-L...")
    hf_hub_download(
        repo_id="BeichenZhang/LongCLIP-L",
        filename="longclip-L.pt",
        cache_dir=HF_CACHE,
    )
    print("  Long-CLIP OK")

    # 3. LongCLIP-GmP ViT-L-14 (zer0int)
    print("[3/4] Downloading LongCLIP-GmP-ViT-L-14 (zer0int)...")
    snapshot_download(
        "zer0int/LongCLIP-GmP-ViT-L-14",
        cache_dir=HF_CACHE,
        ignore_patterns=["*.msgpack", "*.h5", "flax_*"],
    )
    print("  LongCLIP-GmP OK")

    # 4. OpenCLIP ViT-bigG-14
    print("[4/4] Downloading OpenCLIP ViT-bigG-14...")
    if not os.path.exists(OPENCLIP_CACHE):
        bigG, _, _ = open_clip.create_model_and_transforms(
            "ViT-bigG-14", pretrained="laion2b_s39b_b160k")
        torch.save(bigG.state_dict(), OPENCLIP_CACHE)
        del bigG
        print(f"  OpenCLIP saved to {OPENCLIP_CACHE}")
    else:
        print(f"  OpenCLIP already cached at {OPENCLIP_CACHE}")

    # Commit volume để persist
    model_cache.commit()
    print("\n All models downloaded to Volume. Ready for inference!")


@app.local_entrypoint()
def download_models_local():
    """modal run modal_region_edit.py::download_models_local"""
    print("Starting model download on Modal...")
    download_models.remote()
    print("Done! Models cached in Volume 'ultraedit-model-cache'")




def _build_longclip_embeddings(src_caption, tgt_caption, device, dtype):
    """
    Load Long-CLIP + OpenCLIP từ Modal Volume (/cache) — KHÔNG download.
    Cần chạy download_models_local trước lần đầu.
    Load time: Long-CLIP ~5s, OpenCLIP ~15s từ Volume.
    """
    import torch
    import torch.nn as nn

    # ── Long-CLIP ViT-L ───────────────────────────────────────────────────────
    print("    [Long-CLIP] Loading from Volume...")
    sys.path.insert(0, "/repo/UltraEdit/Long-CLIP")
    from model import longclip
    from huggingface_hub import hf_hub_download

    lc_path = hf_hub_download(
        repo_id="BeichenZhang/LongCLIP-L",
        filename="longclip-L.pt",
        cache_dir=HF_CACHE,
        local_files_only=True,
    )
    # Modal Volume không support symlink resolution → tìm blob lớn nhất
    real = os.path.realpath(lc_path)
    if not os.path.exists(real) or os.path.getsize(real) < 1e8:
        # Fallback: tìm file lớn nhất trong blobs/
        blobs_dir = os.path.join(os.path.dirname(os.path.dirname(lc_path)), "blobs")
        if os.path.isdir(blobs_dir):
            files = [os.path.join(blobs_dir, f) for f in os.listdir(blobs_dir)]
            real = max(files, key=os.path.getsize)
    lc_path = real
    lc_model, _ = longclip.load(lc_path, device=device)
    lc_model.eval()
    print(f"    Long-CLIP OK (context_length={lc_model.context_length})")

    # ── OpenCLIP ViT-bigG-14 ──────────────────────────────────────────────────
    print("    [OpenCLIP] Loading from Volume...")
    sys.path.insert(0, "/repo/UltraEdit/Long-CLIP")
    from open_clip_long import factory as open_clip
    if not os.path.exists(OPENCLIP_CACHE):
        raise FileNotFoundError(
            f"OpenCLIP not found at {OPENCLIP_CACHE}. "
            "Run first: modal run modal_region_edit.py::download_models_local")
    bigG, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14")
    ckpt = torch.load(OPENCLIP_CACHE, map_location=device)
    bigG.load_state_dict(ckpt)
    del ckpt

    # KPS positional embedding interpolation
    pos   = bigG.positional_embedding.detach()
    L, D  = pos.shape
    keep  = 20
    new_L = 4 * L - 3 * keep
    new_pos = torch.zeros([new_L, D], dtype=pos.dtype)
    new_pos[:keep] = pos[:keep]
    for i in range(L - 1 - keep):
        k = 4 * i + keep
        new_pos[k]   =  pos[i+keep]
        new_pos[k+1] = (3*pos[i+keep]   +   pos[i+1+keep]) / 4
        new_pos[k+2] = (2*pos[i+keep]   + 2*pos[i+1+keep]) / 4
        new_pos[k+3] = (  pos[i+keep]   + 3*pos[i+1+keep]) / 4
    d = pos[-1] - pos[-2]
    for j in range(4):
        new_pos[new_L - 4 + j] = pos[-1] + j * d / 4
    bigG.positional_embedding = nn.Parameter(new_pos)
    bigG.eval().to(device)
    bigG_tok = open_clip.get_tokenizer("ViT-bigG-14")

    def encode(caption):
        with torch.no_grad():
            # Safe Long-CLIP tokenize — truncate to context_length-2 (BOS+EOS)
            _lc_ctx = lc_model.context_length  # 248
            _sot = longclip._tokenizer.encoder["<|startoftext|>"]
            _eot = longclip._tokenizer.encoder["<|endoftext|>"]
            _raw = longclip._tokenizer.encode(caption)
            _raw = _raw[:_lc_ctx - 2]  # truncate content tokens
            _toks = torch.zeros(1, _lc_ctx, dtype=torch.long)
            _seq  = [_sot] + _raw + [_eot]
            _toks[0, :len(_seq)] = torch.tensor(_seq)
            lc_tok = _toks.to(device)
            emb1   = lc_model.encode_text_full(lc_tok).to(dtype)

            bg_tok = bigG_tok([caption]).to(device)
            emb2   = bigG.encode_text_full(bg_tok).to(dtype)
            pooled = emb2[torch.arange(emb2.shape[0]), bg_tok.argmax(dim=-1)]

            s1, s2 = emb1.shape[1], emb2.shape[1]
            if s1 < s2:
                emb1 = torch.cat([emb1,
                    torch.zeros(1, s2-s1, emb1.shape[2], device=device, dtype=dtype)], 1)
            elif s2 < s1:
                emb2 = torch.cat([emb2,
                    torch.zeros(1, s1-s2, emb2.shape[2], device=device, dtype=dtype)], 1)
            return torch.cat([emb1, emb2], dim=-1), pooled

    src_pe, src_pool = encode(src_caption)
    tgt_pe, tgt_pool = encode(tgt_caption)
    return {
        "prompt_embeds":        torch.cat([src_pe,   tgt_pe],   dim=0),
        "pooled_prompt_embeds": torch.cat([src_pool, tgt_pool], dim=0),
    }


# ═════════════════════════════════════════════════════════════════════════════
# LONGCLIP-GmP  (zer0int/LongCLIP-GmP-ViT-L-14 + CLIP-G zeroed)
# ═════════════════════════════════════════════════════════════════════════════

def _build_longclip_gmp_embeddings(src_caption, tgt_caption, device, dtype):
    """
    Dùng zer0int/LongCLIP-GmP-ViT-L-14 làm CLIP-L (248 tokens),
    zero-out toàn bộ CLIP-G contribution → 'LongCLIP-only' mode.
    Tương đương opendiffusionai/sdxl-longcliponly nhưng dùng GmP weights.

    prompt_embeds        : [2, 248, 2048]  (768 LongCLIP-L + 1280 zeros)
    pooled_prompt_embeds : [2, 1280]       (zeros — addition_embed_type=None nên bị bỏ qua)
    """
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    MODEL_ID   = "zer0int/LongCLIP-GmP-ViT-L-14"
    CLIP_L_DIM = 768
    CLIP_G_DIM = 1280
    MAX_TOKENS = 248

    print(f"    [LongCLIP-GmP] Loading {MODEL_ID}...")
    tokenizer    = CLIPTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE
    ).to(device).to(dtype)
    text_encoder.eval()
    max_pos = text_encoder.config.max_position_embeddings
    print(f"    LongCLIP-GmP OK (max_position_embeddings={max_pos})")

    def encode(caption):
        with torch.no_grad():
            inputs = tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_TOKENS,
            ).to(device)
            out    = text_encoder(**inputs)
            hidden = out.last_hidden_state.to(dtype)           # [1, 248, 768]
            # Zero out CLIP-G contribution
            g_zeros  = torch.zeros(1, MAX_TOKENS, CLIP_G_DIM, device=device, dtype=dtype)
            combined = torch.cat([hidden, g_zeros], dim=-1)    # [1, 248, 2048]
            pooled   = torch.zeros(1, CLIP_G_DIM, device=device, dtype=dtype)
            return combined, pooled

    src_pe, src_pool = encode(src_caption)
    tgt_pe, tgt_pool = encode(tgt_caption)

    del text_encoder
    torch.cuda.empty_cache()

    return {
        "prompt_embeds":        torch.cat([src_pe,   tgt_pe],   dim=0),  # [2, 248, 2048]
        "pooled_prompt_embeds": torch.cat([src_pool, tgt_pool], dim=0),  # [2, 1280]
    }


# ═════════════════════════════════════════════════════════════════════════════
# MODAL FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: model_cache},
    timeout=600,
    memory=16384,
    min_containers=1,   # Giữ 1 container warm → không cold start
                        # Tắt khi không dùng: xóa dòng này hoặc set 0
                        # Chi phí: ~$0.001/phút khi idle (A10G)
)
def run_region_edit(
    # Images (bytes)
    image_bytes:   bytes,
    mf_mask_bytes: bytes,
    mb_mask_bytes: bytes,
    # Captions
    source_caption: str,
    target_caption: str,
    # Params
    soft_mask_value: float = 0.5,
    p2p_threshold:   float = 0.7,
    steps:           int   = 6,
    strength:        float = 0.5,
    guidance_scale:  float = 0.0,
    mask_choice:     str   = "wo_final_layer",  # odd-only=None, all-but-last="wo_final_layer"
    seed:            int   = 42,
    use_long_clip:   bool  = False,
    longclip_encoder: str  = "beichen",  # "beichen" | "zer0int" (GmP, CLIP-G zeroed)
    pipeline_ckpt:   str   = "stabilityai/sdxl-turbo",
    image_size:      int   = 512,
) -> dict:
    """
    Core function chạy trên Modal GPU.
    Nhận ảnh dưới dạng bytes, trả về dict chứa output images dưới dạng bytes.
    """
    import torch
    import random
    import numpy as np
    import warnings
    import logging
    from PIL import Image

    warnings.filterwarnings("ignore")
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Silence "cross_attention_kwargs not expected" từ P2PCrossAttnProcessor
    import logging as _logging
    _orig = _logging.Logger.warning
    def _filtered(self, msg, *args, **kwargs):
        if "cross_attention_kwargs" in str(msg) and "not expected" in str(msg):
            return
        _orig(self, msg, *args, **kwargs)
    _logging.Logger.warning = _filtered

    device = "cuda"
    dtype  = torch.float16

    print(f"\n{'='*55}")
    print(f"  UltraEdit on Modal — GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'='*55}")

    # ── Setup repo paths ───────────────────────────────────────────────────────
    sys.path.insert(0, "/repo/UltraEdit/diffusers/src")
    sys.path.insert(0, "/repo/UltraEdit/data_generation")
    sys.path.insert(0, "/repo/UltraEdit/Long-CLIP")

    from sdxl_p2p_pipeline import Prompt2PromptInpaintPipeline

    # ── Load images từ bytes ───────────────────────────────────────────────────
    def bytes_to_pil(b, mode="RGB"):
        img = Image.open(io.BytesIO(b)).convert(mode)
        return img

    def resize_center(img, size):
        w, h = img.size
        if w > h:
            img = img.resize((int(size*w/h), size), Image.LANCZOS)
        else:
            img = img.resize((size, int(size*h/w)), Image.LANCZOS)
        w, h = img.size
        return img.crop(((w-size)//2, (h-size)//2, (w+size)//2, (h+size)//2))

    image   = resize_center(bytes_to_pil(image_bytes),   image_size)
    mf_mask = bytes_to_pil(mf_mask_bytes, "L").resize((image_size, image_size), Image.NEAREST)
    mb_mask = bytes_to_pil(mb_mask_bytes, "L").resize((image_size, image_size), Image.NEAREST)

    # ── Load Pipeline ──────────────────────────────────────────────────────────
    print(f"\n[1] Loading pipeline: {pipeline_ckpt}")
    pipe = Prompt2PromptInpaintPipeline.from_pretrained(
        pipeline_ckpt,
        torch_dtype=dtype,
        variant="fp16",
        cache_dir="/cache/huggingface",
    ).to(device)
    pipe.unet.config.addition_embed_type = None
    pipe.set_progress_bar_config(disable=False)
    print(f"  OK — UNet in_channels: {pipe.unet.config.in_channels}")

    # ── Captions ───────────────────────────────────────────────────────────────
    print(f"\n[2] Captions ({('Long-CLIP 248' if use_long_clip else 'CLIP 77')} tokens)")
    src_cap = source_caption
    tgt_cap = target_caption

    longclip_embeds = None
    if use_long_clip:
        try:
            if longclip_encoder == "zer0int":
                print("  Building LongCLIP-GmP embeddings (CLIP-G zeroed)...")
                longclip_embeds = _build_longclip_gmp_embeddings(
                    src_cap, tgt_cap, device=device, dtype=dtype)
            else:  # "beichen" (default)
                print("  Building Long-CLIP embeddings (Beichen + OpenCLIP bigG)...")
                longclip_embeds = _build_longclip_embeddings(
                    src_cap, tgt_cap, device=device, dtype=dtype)
            print(f"  prompt_embeds: {tuple(longclip_embeds['prompt_embeds'].shape)}")
        except Exception as e:
            print(f"  [FALLBACK] Long-CLIP failed: {e} → CLIP 77 tokens")
            longclip_embeds = None

    if longclip_embeds is None:
        # Truncate to 75 tokens
        tokens = pipe.tokenizer.encode(src_cap)
        if len(tokens) > 75:
            src_cap = pipe.tokenizer.decode(
                pipe.tokenizer.encode(src_cap)[:75], skip_special_tokens=True)
        tokens = pipe.tokenizer.encode(tgt_cap)
        if len(tokens) > 75:
            tgt_cap = pipe.tokenizer.decode(
                pipe.tokenizer.encode(tgt_cap)[:75], skip_special_tokens=True)

    # ── P2P kwargs ─────────────────────────────────────────────────────────────
    diff = [b for a, b in zip(src_cap.split(), tgt_cap.split()) if a != b]
    if len(diff) > 0 and len(src_cap.split()) == len(tgt_cap.split()):
        edit_type = "replace"
        key       = " ".join(diff) if len(diff) <= 5 else None
        n_cross   = {"default_": 1.0, key: p2p_threshold} if key else p2p_threshold
    else:
        edit_type = "refine"
        n_cross   = p2p_threshold
    print(f"  P2P edit_type: {edit_type} | diff: {diff[:5]}")
    cross_attn_kwargs = {
        "edit_type":       edit_type,
        "n_self_replace":  p2p_threshold,
        "n_cross_replace": n_cross,
        "prompts":         [src_cap, tgt_cap],
        "max_num_words":   248 if longclip_embeds is not None else 77,
    }

    # ── Generate ───────────────────────────────────────────────────────────────
    print(f"\n[3] Generating (steps={steps}, strength={strength}, s={soft_mask_value})")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    call_kwargs = dict(
        image          = image,
        mask_image     = mf_mask,
        temp_mask      = mb_mask,
        soft_mask      = soft_mask_value,
        mask_choice    = mask_choice,
        num_inference_steps = steps,
        strength       = strength,
        guidance_scale = guidance_scale,
        cross_attention_kwargs = cross_attn_kwargs,
        output_type    = "pil",
    )
    if longclip_embeds is not None:
        call_kwargs["prompt"] = None        # None bắt buộc khi dùng prompt_embeds
        call_kwargs["prompt_embeds"] = longclip_embeds["prompt_embeds"]
        call_kwargs["pooled_prompt_embeds"] = longclip_embeds["pooled_prompt_embeds"]
    else:
        call_kwargs["prompt"] = [src_cap, tgt_cap]

    out = pipe(**call_kwargs).images
    print("  Generation done ✓")

    # ── Encode output về bytes để trả về ──────────────────────────────────────
    def pil_to_bytes(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return {
        "source_recon": pil_to_bytes(out[0]),
        "target":       pil_to_bytes(out[1]),
    }


# ═════════════════════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT  (chạy từ máy local, gửi lên Modal)
# ═════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    image_path:     str  = "mask_output/input_resized.png",
    mf_mask:        str  = "mask_output/Mf_fine_grained.png",
    mb_mask:        str  = "mask_output/Mb_bounding_box.png",
    source_caption: str  = "white monster face graffiti on brick wall behind fence",
    target_caption: str  = "blue monster face graffiti on brick wall behind fence",
    soft_mask_value: float = 0.5,
    p2p_threshold:  float = 0.7,
    steps:          int   = 6,
    strength:       float = 0.5,
    seed:           int   = 42,
    use_long_clip:  bool  = False,
    output_dir:     str   = "modal_output",
    pipeline_ckpt:  str   = "stabilityai/sdxl-turbo",
):
    """
    Local entrypoint: đọc file từ máy local, gửi lên Modal, download kết quả về.
    """
    from PIL import Image

    print(f"\n{'='*55}")
    print(f"  UltraEdit Modal Runner")
    print(f"  Image:  {image_path}")
    print(f"  Mf:     {mf_mask}")
    print(f"  Mb:     {mb_mask}")
    print(f"  Encoder: {'Long-CLIP 248 tokens' if use_long_clip else 'CLIP 77 tokens'}")
    print(f"{'='*55}\n")

    # Đọc files local
    def read_bytes(path):
        with open(path, "rb") as f:
            return f.read()

    print("Reading local files...")
    image_bytes   = read_bytes(image_path)
    mf_bytes      = read_bytes(mf_mask)
    mb_bytes      = read_bytes(mb_mask)

    # Gửi lên Modal và chạy
    print("Sending to Modal GPU...")
    result = run_region_edit.remote(
        image_bytes    = image_bytes,
        mf_mask_bytes  = mf_bytes,
        mb_mask_bytes  = mb_bytes,
        source_caption = source_caption,
        target_caption = target_caption,
        soft_mask_value = soft_mask_value,
        p2p_threshold  = p2p_threshold,
        steps          = steps,
        strength       = strength,
        seed           = seed,
        use_long_clip  = use_long_clip,
        pipeline_ckpt  = pipeline_ckpt,
    )

    # Download kết quả về local
    os.makedirs(output_dir, exist_ok=True)
    tag = "_longclip" if use_long_clip else "_clip77"
    prefix = f"s{soft_mask_value}_str{strength}_seed{seed}{tag}"

    src_path = os.path.join(output_dir, f"{prefix}_src_recon.png")
    tgt_path = os.path.join(output_dir, f"{prefix}_target.png")

    Image.open(io.BytesIO(result["source_recon"])).save(src_path)
    Image.open(io.BytesIO(result["target"])).save(tgt_path)

    print(f"\n  Downloaded results to {output_dir}/")
    print(f"  {prefix}_src_recon.png")
    print(f"  {prefix}_target.png   ← TARGET")
    print(f"\n  Done!")