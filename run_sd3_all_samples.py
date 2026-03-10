#!/usr/bin/env python3
"""
Chạy tất cả eval_data samples qua SD3 UltraEdit với target_caption_long.

Usage:
    python3 run_sd3_all_samples.py
    python3 run_sd3_all_samples.py --samples sample_01 sample_03
    python3 run_sd3_all_samples.py --out-dir modal_output_sd3/long_caption
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

SAMPLES = [
    "sample_01",
    "sample_02",
    "sample_03",
    "sample_04",
    "sample_05",
    "sample_ue",
]

def load_meta(sample: str) -> dict:
    meta_path = ROOT / "eval_data" / sample / "metadata.json"
    with open(meta_path) as f:
        return json.load(f)

def run_sample(sample: str, out_dir: str, steps: int, cfg: float, img_cfg: float, seed: int) -> bool:
    meta        = load_meta(sample)
    edit_prompt = meta.get("target_caption_long", "").strip()
    src_prompt  = meta.get("source_caption_long",  "").strip()

    if not edit_prompt:
        print(f"  [SKIP] {sample}: không có target_caption_long")
        return False

    image_path = ROOT / "eval_data" / sample / "image.png"
    mf_mask    = ROOT / "eval_data" / sample / "Mf_mask.png"

    if not image_path.exists():
        print(f"  [SKIP] {sample}: thiếu image.png")
        return False
    if not mf_mask.exists():
        print(f"  [SKIP] {sample}: thiếu Mf_mask.png")
        return False

    sample_out = str(Path(out_dir) / sample)

    print(f"\n{'='*60}")
    print(f"  Sample : {sample}")
    print(f"  Prompt : {edit_prompt[:120]}{'...' if len(edit_prompt) > 120 else ''}")
    print(f"  Out    : {sample_out}")
    print(f"{'='*60}")

    cmd = [
        "modal", "run", "modal_sd3_edit.py",
        "--image-path",    str(image_path),
        "--mf-mask",       str(mf_mask),
        "--edit-prompt",   edit_prompt,
        "--source-prompt", src_prompt,
        "--output-dir",    sample_out,
        "--steps",         str(steps),
        "--guidance-scale", str(cfg),
        "--image-guidance", str(img_cfg),
        "--seed",           str(seed),
    ]

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [ERROR] {sample} failed (exit {result.returncode})")
        return False

    print(f"  [OK] {sample} → {sample_out}/")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",       nargs="+", default=SAMPLES,
                        help="Danh sách sample cần chạy (mặc định: tất cả)")
    parser.add_argument("--out-dir",       default="modal_output_sd3",
                        help="Thư mục lưu kết quả (mặc định: modal_output_sd3)")
    parser.add_argument("--steps",         type=int,   default=28)
    parser.add_argument("--guidance-scale",type=float, default=7.0)
    parser.add_argument("--image-guidance",type=float, default=1.5)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    print(f"\n🚀 SD3 UltraEdit — long target caption run")
    print(f"   Samples  : {args.samples}")
    print(f"   Out dir  : {args.out_dir}")
    print(f"   Steps    : {args.steps} | text_cfg={args.guidance_scale} | img_cfg={args.image_guidance} | seed={args.seed}\n")

    ok, fail = [], []
    for sample in args.samples:
        success = run_sample(
            sample,
            out_dir=args.out_dir,
            steps=args.steps,
            cfg=args.guidance_scale,
            img_cfg=args.image_guidance,
            seed=args.seed,
        )
        (ok if success else fail).append(sample)

    print(f"\n{'='*60}")
    print(f"  ✅ Thành công : {ok}")
    print(f"  ❌ Thất bại  : {fail}")
    print(f"{'='*60}\n")
    return 0 if not fail else 1

if __name__ == "__main__":
    sys.exit(main())
