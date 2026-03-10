"""
GT-style inpainting: dùng StableDiffusionXLInpaintPipeline chuẩn (batch=1)
giống cách UltraEdit dataset tạo ground truth.
  - model: diffusers/stable-diffusion-xl-1.0-inpainting-0.1  (9-ch UNet)
  - steps: 20, guidance_scale: 7.5, strength: 1.0
  - background: pixel-perfect copy từ source
"""
import torch
import numpy as np
from PIL import Image
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diffusers", "src"))

from diffusers import StableDiffusionXLInpaintPipeline

# ── params ────────────────────────────────────────────────────────────────────
IMAGE    = "eval_data/sample_ue/image.png"
MF_MASK  = "eval_data/sample_ue/Mf_mask.png"
PROMPT   = "Two giraffes are next to a colourful rainbow tree."
NEG_PROMPT = ""
STEPS    = 20
STRENGTH = 1.0
CFG      = 7.5
SEED     = 42
SIZE     = 512
OUT_DIR  = "region_output_gt_style"
CKPT     = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"  GT-style Inpainting (standard, batch=1)")
print(f"  Model   : {CKPT}")
print(f"  Steps   : {STEPS}  |  CFG: {CFG}  |  strength: {STRENGTH}")
print(f"{'='*60}\n")

# ── load image + mask ─────────────────────────────────────────────────────────
image   = Image.open(IMAGE).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
mask    = Image.open(MF_MASK).convert("L").resize((SIZE, SIZE), Image.NEAREST)

print(f"[1] Loading pipeline (bfloat16 CPU — ~5.1GB RAM)...")
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    CKPT,
    torch_dtype=torch.bfloat16,
).to("cpu")
pipe.set_progress_bar_config(disable=False)
print(f"    UNet in_channels: {pipe.unet.config.in_channels}")

print(f"\n[2] Generating...")
torch.manual_seed(SEED)
result = pipe(
    prompt              = PROMPT,
    negative_prompt     = NEG_PROMPT,
    image               = image,
    mask_image          = mask,
    num_inference_steps = STEPS,
    strength            = STRENGTH,
    guidance_scale      = CFG,
    height              = SIZE,
    width               = SIZE,
    output_type         = "pil",
).images[0]

# ── save ──────────────────────────────────────────────────────────────────────
out_path  = os.path.join(OUT_DIR, f"gt_style_seed{SEED}.png")
comp_path = os.path.join(OUT_DIR, f"gt_style_comparison.png")

result.save(out_path)
print(f"\n[3] Saved: {out_path}")

# comparison: Source | Mask | Output | GT
gt = Image.open("eval_data/sample_ue/gt_edited.png").convert("RGB").resize((SIZE, SIZE))
mask_rgb = mask.convert("RGB")
comp = Image.new("RGB", (SIZE * 4, SIZE), (20, 20, 20))
for i, (img, label) in enumerate([
    (image,    "Source"),
    (mask_rgb, "Mask (Mf)"),
    (result,   "GT-style output"),
    (gt,       "Dataset GT"),
]):
    comp.paste(img, (i * SIZE, 0))
    from PIL import ImageDraw
    d = ImageDraw.Draw(comp)
    d.rectangle([(i*SIZE+2, 2), (i*SIZE+len(label)*8+8, 22)], fill="black")
    d.text((i*SIZE+5, 4), label, fill="white")

comp.save(comp_path)
print(f"    Comparison: {comp_path}")

# PSNR vs GT
arr_out = np.array(result).astype(float)
arr_gt  = np.array(gt).astype(float)
mse     = np.mean((arr_out - arr_gt) ** 2)
psnr    = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
mask_np = (np.array(mask) > 127)
mse_fg  = np.mean((arr_out[mask_np] - arr_gt[mask_np]) ** 2)
psnr_fg = 10 * np.log10(255**2 / mse_fg) if mse_fg > 0 else float("inf")
print(f"\n    PSNR_full : {psnr:.2f} dB")
print(f"    PSNR_fg   : {psnr_fg:.2f} dB")
print(f"\n{'='*60}  Done!\n")
