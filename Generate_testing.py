"""
Region-based Image Editing using Soft Mask (Prompt2PromptInpaintPipeline)
"""

import sys
import os
import argparse
import random
import torch
import numpy as np
from PIL import Image

# Thêm path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data_generation"))
from sdxl_p2p_pipeline import Prompt2PromptInpaintPipeline
from util import create_controller

def load_and_resize(image_path, target_size=512):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    if width > height:
        new_width = int(target_size * width / height)
        img = img.resize((new_width, target_size), Image.LANCZOS)
    else:
        new_height = int(target_size * height / width)
        img = img.resize((target_size, new_height), Image.LANCZOS)
    width, height = img.size
    left = (width - target_size) // 2
    top = (height - target_size) // 2
    return img.crop((left, top, left + target_size, top + target_size))

def compare_prompts(prompt1, prompt2):
    return [w2 for w1, w2 in zip(prompt1.split(), prompt2.split()) if w1 != w2]

def main():
    parser = argparse.ArgumentParser(description="P2P Inpainting with Soft Mask")
    parser.add_argument("--image", required=True)
    parser.add_argument("--soft_mask", required=True, help="Soft mask (Ms)")
    parser.add_argument("--source_caption", required=True)
    parser.add_argument("--target_caption", required=True)
    parser.add_argument("--target_object", default=None)
    parser.add_argument("--output_dir", default="region_output")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pipeline_ckpt", default="stabilityai/sdxl-turbo")
    parser.add_argument("--p2p_threshold", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Device
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} | Dtype: {dtype}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Inpainting Pipeline from: {args.pipeline_ckpt}")
    pipe = Prompt2PromptInpaintPipeline.from_pretrained(
        args.pipeline_ckpt, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    ).to(device)

    # SDXL-Turbo configs
    pipe.unet.config.addition_embed_type = None
    pipe.set_progress_bar_config(disable=False)

    image = load_and_resize(args.image)
    # Load soft mask - note that Prompt2PromptInpaintPipeline accepts PIL mask and converts to [0, 1] internally
    # Wait, the pipeline expects 'L' mode. Let's make sure it's correct.
    soft_mask_img = Image.open(args.soft_mask).convert("L")
    soft_mask_img = soft_mask_img.resize((512, 512), Image.BILINEAR)

    print(f"\nSource: {args.source_caption[:50]}...")
    print(f"Target: {args.target_caption[:50]}...")

    # Truncate to ~70 words to roughly avoid 77 token limit error in util.py
    src_words = args.source_caption.split()
    if len(src_words) > 65:
        args.source_caption = " ".join(src_words[:65])
        print(f"  [Warning] Truncated source caption to 65 words: {args.source_caption[-30:]}")
        
    tgt_words = args.target_caption.split()
    if len(tgt_words) > 65:
        args.target_caption = " ".join(tgt_words[:65])
        print(f"  [Warning] Truncated target caption to 65 words: {args.target_caption[-30:]}")

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Determine P2P edit args
    diff_words = compare_prompts(args.source_caption, args.target_caption)
    if len(diff_words) > 0 and len(args.source_caption.split()) == len(args.target_caption.split()):
        edit_type = "replace"
        cross_replace = " ".join(diff_words) if len(diff_words) <= 5 else args.p2p_threshold
        n_cross_replace = (
            {"default_": 1.0, str(cross_replace): args.p2p_threshold}
            if isinstance(cross_replace, str)
            else args.p2p_threshold
        )
    else:
        edit_type = "refine"
        n_cross_replace = args.p2p_threshold

    cross_attention_kwargs = {
        "edit_type": edit_type,
        "n_self_replace": args.p2p_threshold,
        "n_cross_replace": n_cross_replace,
    }

    # Generate
    print("\nRunning generation...")
    # NOTE: Since we pass soft_mask directly as `mask_image`, the loop does:
    # latents = (1 - mask_image) * original + mask_image * new
    # which is exactly alternating blending if we don't pass temp_mask.
    out = pipe(
        prompt=[args.source_caption, args.target_caption],
        image=image,
        mask_image=soft_mask_img,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        cross_attention_kwargs=cross_attention_kwargs,
        output_type="pil"
    ).images

    # Save
    prefix = f"edited_{args.seed}"
    out[0].save(os.path.join(args.output_dir, f"{prefix}_source_recon.png"))
    out[1].save(os.path.join(args.output_dir, f"{prefix}_target.png"))

    # Side-by-side
    w, h = image.size
    comp = Image.new("RGB", (w * 3, h))
    comp.paste(image, (0, 0))
    comp.paste(soft_mask_img.convert("RGB"), (w, 0))
    comp.paste(out[1], (w * 2, 0))
    
    comp_path = os.path.join(args.output_dir, f"{prefix}_comparison.png")
    comp.save(comp_path)
    print(f"\nSaved results to {args.output_dir}/")
    print(f"Comparison: {comp_path}")

if __name__ == "__main__":
    main()