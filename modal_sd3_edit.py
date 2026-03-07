"""
Modal Deployment — UltraEdit SD3 Region-based Edit
===================================================
Dùng BleachNick/SD3_UltraEdit_w_mask:
  - StableDiffusion3InstructPix2PixPipeline (diffusers fork)
  - Text encoders: CLIP-L (77t) + CLIP-G (77t) + T5-XXL (256t)
  - mask_img: truyền trực tiếp vào pipeline, không cần P2P tricks
  - CFG guidance_scale=7.0 + image_guidance_scale=1.5

Lưu ý fork bug: pipeline fork set tokenizer_max_length = CLIP.model_max_length = 77,
vô tình giới hạn T5 xuống 77. File này patch lại T5 về 256 sau khi load.

Ưu điểm so với SDXL-Turbo + Long-CLIP:
  ✅ SD3 DiT architecture: cross-attention không bị dilution như SDXL UNet
  ✅ T5-XXL 256 tokens vs Long-CLIP 248 tokens, nhưng T5 semantic representation
     phong phú hơn CLIP rất nhiều (T5 language model vs CLIP contrastive)
  ✅ Mask-aware natively: model được finetune với mask → không cần P2P injection
  ✅ guidance_scale=7.0 (thay vì SDXL-Turbo guidance=0) → kiểm soát tốt hơn

Setup:
  modal run modal_sd3_edit.py::download_models_sd3
  modal run modal_sd3_edit.py --image-path ... --source-caption ... --edit-prompt ...
"""

import modal
import os
import io
import sys

# ── Modal App ──────────────────────────────────────────────────────────────────
app = modal.App("ultraedit-sd3-edit")

# ── Docker Image ───────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev",
    )
    .pip_install("numpy<2")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # transformers>=4.42 có T5TokenizerFast đầy đủ
        "transformers>=4.42.0",
        "accelerate>=0.24.0",
        "huggingface_hub>=0.19.0",
        "safetensors",
        "sentencepiece",    # T5 tokenizer cần
    )
    .pip_install(
        "Pillow>=9.0.0",
        "scipy",
        "ftfy",
        "regex",
        "tqdm",
        # requests cần có pip metadata (debian_slim cài system package, không có metadata)
        # diffusers fork check metadata tại import time → fail nếu thiếu
        "requests",
        "filelock",
        "packaging",
    )
    # Clone UltraEdit repo để lấy diffusers fork có SD3 InstructPix2Pix pipeline
    # Overlay pipeline_loading_utils.py đã patch (fix FLAX_WEIGHTS_NAME removed in transformers>=4.44)
    .run_commands(
        "git clone https://github.com/HaozheZhao/UltraEdit.git /repo/UltraEdit",
    )
    .add_local_file(
        "diffusers/src/diffusers/pipelines/pipeline_loading_utils.py",
        "/repo/UltraEdit/diffusers/src/diffusers/pipelines/pipeline_loading_utils.py",
        copy=True,  # copy=True bắt buộc khi có run_commands sau add_local_*
    )
    .run_commands(
        "cd /repo/UltraEdit/diffusers && pip install -e . --no-deps 2>/dev/null || true",
    )
)

# ── Volume ─────────────────────────────────────────────────────────────────────
model_cache = modal.Volume.from_name(
    "ultraedit-sd3-cache", create_if_missing=True)

CACHE_DIR = "/cache"
HF_CACHE  = "/cache/huggingface"
GPU_CONFIG = "A10G"   # SD3 cần ~18GB VRAM → A10G (24GB) OK


# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOAD MODELS
# modal run modal_sd3_edit.py::download_models_sd3
# ═════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={CACHE_DIR: model_cache},
    timeout=7200,  # T5-XXL lớn ~10GB → cần nhiều thời gian
    cpu=4,
    memory=32768,
)
def download_sd3_models():
    """Download BleachNick/SD3_UltraEdit_w_mask vào Volume."""
    from huggingface_hub import snapshot_download

    os.makedirs(HF_CACHE, exist_ok=True)

    print("[1/1] Downloading SD3_UltraEdit_w_mask (~16GB)...")
    print("  Includes: T5-XXL (10GB) + CLIP-L + CLIP-G + DiT Transformer + VAE")
    snapshot_download(
        "BleachNick/SD3_UltraEdit_w_mask",
        cache_dir=HF_CACHE,
        ignore_patterns=["*.msgpack", "*.h5", "flax_*"],
    )
    print("  SD3_UltraEdit OK")

    model_cache.commit()
    print("\nAll models downloaded. Ready for inference!")


def download_models_sd3():
    """modal run modal_sd3_edit.py::download_models_sd3"""
    print("Starting SD3 model download on Modal...")
    download_sd3_models.remote()
    print("Done! Models cached in Volume 'ultraedit-sd3-cache'")


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: model_cache},
    timeout=600,
    memory=32768,
    min_containers=0,  # Không giữ warm — SD3 load ~40s, tốn tiền idle
)
def run_sd3_edit(
    image_bytes:     bytes,
    mask_bytes:      bytes,        # Mf SAM mask (white=edit region)
    edit_prompt:     str,          # Instruction-style: "Change X to Y"
    source_prompt:   str   = "",   # Source description (optional, cải thiện quality)
    steps:           int   = 28,   # SD3 default
    guidance_scale:  float = 7.0,  # Text CFG
    image_guidance:  float = 1.5,  # Image CFG (InstructPix2Pix)
    seed:            int   = 42,
    image_size:      int   = 512,
) -> dict:
    """
    SD3 UltraEdit inference trên Modal GPU.

    Khác với SDXL pipeline:
      - Không có P2P attention injection
      - Không có SDEdit strength
      - mask_img truyền trực tiếp → model học mask-aware từ training
      - T5-XXL xử lý caption dài natively (256 tokens)
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

    device = "cuda"
    dtype  = torch.bfloat16  # SD3 dùng bfloat16, không phải float16

    print(f"\n{'='*55}")
    print(f"  UltraEdit SD3 on Modal — GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'='*55}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    sys.path.insert(0, "/repo/UltraEdit/diffusers/src")

    from diffusers import StableDiffusion3InstructPix2PixPipeline

    # ── Load images ───────────────────────────────────────────────────────────
    def bytes_to_pil(b, mode="RGB"):
        return Image.open(io.BytesIO(b)).convert(mode)

    def resize_center(img, size):
        w, h = img.size
        if w > h:
            img = img.resize((int(size*w/h), size), Image.LANCZOS)
        else:
            img = img.resize((size, int(size*h/w)), Image.LANCZOS)
        w, h = img.size
        return img.crop(((w-size)//2, (h-size)//2, (w+size)//2, (h+size)//2))

    source_image = resize_center(bytes_to_pil(image_bytes), image_size)
    # mask: đọc dạng grayscale, resize, rồi convert sang RGB
    # VaeImageProcessor.preprocess cần 3-channel để encode qua VAE đúng
    _mask_gray = bytes_to_pil(mask_bytes, "L").resize(
        (image_size, image_size), Image.NEAREST)
    mask_image = _mask_gray.convert("RGB")

    # ── Load Pipeline ─────────────────────────────────────────────────────────
    print(f"\n[1] Loading SD3_UltraEdit_w_mask...")
    pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
        "BleachNick/SD3_UltraEdit_w_mask",
        torch_dtype=dtype,
        cache_dir=HF_CACHE,
    ).to(device)

    # ── Patch fork bug: tokenizer_max_length dùng CLIP.model_max_length (77)
    # vô tình giới hạn T5 xuống 77. Cần tách riêng: CLIP=77, T5=256.
    # Giải pháp: monkey-patch _get_t5_prompt_embeds để dùng T5_MAX=256 cố định,
    # giữ nguyên tokenizer_max_length=77 cho CLIP.
    import types

    T5_MAX = 256

    def _patched_get_t5_prompt_embeds(self, prompt=None, num_images_per_prompt=1, device=None, dtype=None):
        import torch as _torch
        device = device or self._execution_device
        dtype  = dtype  or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return _torch.zeros(
                (batch_size, T5_MAX, self.transformer.config.joint_attention_dim),
                device=device, dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=T5_MAX,      # ← 256, bỏ qua tokenizer_max_length (77)
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder_3(text_inputs.input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_3.dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds

    pipe._get_t5_prompt_embeds = types.MethodType(_patched_get_t5_prompt_embeds, pipe)
    print(f"  [patch] T5 tokenizer_max_length: 77 (fork default) → {T5_MAX} (monkey-patched)")

    pipe.set_progress_bar_config(disable=False)
    print(f"  OK — Transformer params: {sum(p.numel() for p in pipe.transformer.parameters())/1e9:.1f}B")

    # ── Generate ──────────────────────────────────────────────────────────────
    # SD3 InstructPix2Pix:
    #   prompt        = edit instruction ("Change white to blue")
    #   image         = source image
    #   mask_img      = SAM mask (white region = edit)
    #   guidance_scale         = text CFG (7.0 recommended)
    #   image_guidance_scale   = image preservation (1.5 recommended)
    print(f"\n[2] Generating...")
    print(f"  Edit: \"{edit_prompt[:100]}{'...' if len(edit_prompt)>100 else ''}\"")
    print(f"  Steps={steps}, text_cfg={guidance_scale}, img_cfg={image_guidance}")
    print(f"  T5-XXL tokens: ~{min(len(edit_prompt.split()), T5_MAX)} word-pieces (max {T5_MAX})")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    result = pipe(
        prompt               = edit_prompt,
        image                = source_image,
        mask_img             = mask_image,
        num_inference_steps  = steps,
        guidance_scale       = guidance_scale,
        image_guidance_scale = image_guidance,
        generator            = torch.Generator(device).manual_seed(seed),
        output_type          = "pil",
    ).images[0]

    print("  Generation done ✓")

    def pil_to_bytes(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return {
        "target": pil_to_bytes(result),
    }


# ═════════════════════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    image_path:     str   = "mask_output/input_resized.png",
    mf_mask:        str   = "mask_output/Mf_fine_grained.png",
    edit_prompt:    str   = "Change the white monster face graffiti to blue",
    source_prompt:  str   = "",
    steps:          int   = 28,
    guidance_scale: float = 7.0,
    image_guidance: float = 1.5,
    seed:           int   = 42,
    output_dir:     str   = "modal_output_sd3",
    image_size:     int   = 512,
    download_only:  bool  = False,
):
    """
    Usage:
      # Download models trước (1 lần):
      modal run modal_sd3_edit.py::main --download-only

      # Inference:
      modal run modal_sd3_edit.py \\
        --image-path mask_output/input_resized.png \\
        --mf-mask mask_output/Mf_fine_grained.png \\
        --edit-prompt "Change the white monster face graffiti to blue"
    """
    from PIL import Image

    if download_only:
        print("Downloading SD3 models to Volume...")
        download_sd3_models.remote()
        print("Done!")
        return

    print(f"\n{'='*55}")
    print(f"  UltraEdit SD3 Runner")
    print(f"  Image:       {image_path}")
    print(f"  Mask:        {mf_mask}")
    print(f"  Edit prompt: {edit_prompt[:80]}...")
    print(f"  Steps: {steps} | text_cfg: {guidance_scale} | img_cfg: {image_guidance}")
    print(f"{'='*55}\n")

    def read_bytes(path):
        with open(path, "rb") as f:
            return f.read()

    print("Reading local files...")
    image_bytes = read_bytes(image_path)
    mask_bytes  = read_bytes(mf_mask)

    print("Sending to Modal GPU...")
    result = run_sd3_edit.remote(
        image_bytes    = image_bytes,
        mask_bytes     = mask_bytes,
        edit_prompt    = edit_prompt,
        source_prompt  = source_prompt,
        steps          = steps,
        guidance_scale = guidance_scale,
        image_guidance = image_guidance,
        seed           = seed,
        image_size     = image_size,
    )

    os.makedirs(output_dir, exist_ok=True)
    prefix   = f"sd3_cfg{guidance_scale}_imgcfg{image_guidance}_seed{seed}"
    tgt_path = os.path.join(output_dir, f"{prefix}_target.png")

    Image.open(io.BytesIO(result["target"])).save(tgt_path)

    print(f"\n  Saved to {output_dir}/")
    print(f"  {prefix}_target.png  ← TARGET")
    print(f"\n  Done!")
