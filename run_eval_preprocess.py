"""
run_eval_preprocess.py
======================
Step 1 của evaluation pipeline:
  - Resize ảnh gốc → 512×512, save vào eval_data/sample_NN/image.png
  - Sinh Mf (SAM fine-grained) + Mb (bounding box) mask cho mỗi sample
    bằng Grounded-SAM với edit_region làm text query

Dùng cách nào sinh mask:
  A) --mask-mode grounded-sam  : dùng local Grounded-SAM (cần GPU + setup đầy đủ)
  B) --mask-mode manual        : copy mask từ mask_output/ (debug/demo nhanh)
  C) --mask-mode skip          : bỏ qua mask, chỉ resize ảnh (dùng khi tự vẽ mask sau)

Usage:
  python run_eval_preprocess.py --mask-mode skip
  python run_eval_preprocess.py --mask-mode manual --manual-mf mask_output/Mf_fine_grained.png --manual-mb mask_output/Mb_bounding_box.png
  python run_eval_preprocess.py --mask-mode grounded-sam
"""

import os
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent
RAW_DATA    = REPO_ROOT / "Raw_Data(images with cation)"
EVAL_DATA   = REPO_ROOT / "eval_data"
SAMPLES     = sorted([d for d in EVAL_DATA.iterdir() if d.is_dir() and d.name.startswith("sample_")])

IMAGE_SIZE  = 512  # resize target


def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Resize ngắn cạnh = size, center crop thành square."""
    w, h = img.size
    if w < h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_w, new_h = int(size * w / h), size
    img = img.resize((new_w, new_h), Image.LANCZOS)
    w, h = img.size
    left = (w - size) // 2
    top  = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


# ── Mode A: Grounded-SAM ───────────────────────────────────────────────────────
def run_grounded_sam(image_path: Path, text_query: str, out_dir: Path):
    """
    Dùng Grounded-SAM để sinh mask từ text_query.
    Cần: GroundingDINO + SAM weights đã setup tại data_generation/Grounded-Segment-Anything/
    """
    import sys
    gsa_dir = REPO_ROOT / "data_generation" / "Grounded-Segment-Anything"
    sys.path.insert(0, str(gsa_dir))

    import torch
    import supervision as sv
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    from segment_anything import sam_model_registry, SamPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── GroundingDINO ──────────────────────────────────────────────────────────
    # Weights thực tế nằm trong model/ subfolder
    dino_cfg    = gsa_dir / "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
    dino_ckpt   = gsa_dir / "model/grounding_dino/groundingdino_swinb_cogcoor.pth"
    sam_ckpt    = gsa_dir / "model/sam/sam_vit_h_4b8939.pth"

    if not dino_ckpt.exists() or not sam_ckpt.exists():
        missing = [p for p in [dino_ckpt, sam_ckpt] if not p.exists()]
        print(f"  [WARN] Grounded-SAM weights not found:")
        for p in missing:
            print(f"    {p}")
        print(f"  [WARN] Skipping mask generation for {image_path.name}")
        return False

    dino_model = load_model(str(dino_cfg), str(dino_ckpt))
    image_source, image_tensor = load_image(str(image_path))

    BOX_THRESHOLD  = 0.35
    TEXT_THRESHOLD = 0.25

    boxes, logits, phrases = predict(
        model     = dino_model,
        image     = image_tensor,
        caption   = text_query,
        box_threshold  = BOX_THRESHOLD,
        text_threshold = TEXT_THRESHOLD,
    )

    print(f"  GroundingDINO found {len(boxes)} box(es) for query: \"{text_query}\"")

    # ── SAM ───────────────────────────────────────────────────────────────────
    sam = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt)).to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_source)

    H, W, _ = image_source.shape

    # GroundingDINO returns normalized cxcywh → convert to pixel xyxy
    boxes_cxcywh = boxes * torch.tensor([W, H, W, H], dtype=torch.float32)
    cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    boxes_xyxy = torch.stack([
        (cx - bw / 2).clamp(min=0),
        (cy - bh / 2).clamp(min=0),
        (cx + bw / 2).clamp(max=W),
        (cy + bh / 2).clamp(max=H),
    ], dim=1)

    # Bounding box mask Mb: union of all boxes (xyxy pixel coords)
    mb_mask = np.zeros((H, W), dtype=np.uint8)
    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        mb_mask[y1:y2, x1:x2] = 255

    # Fine-grained mask Mf: SAM segmentation
    if len(boxes_xyxy) > 0:
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_xyxy.to(device), image_source.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        mf_mask = (masks[0, 0].cpu().numpy() * 255).astype(np.uint8)
        for i in range(1, len(masks)):
            mf_mask = np.maximum(mf_mask, (masks[i, 0].cpu().numpy() * 255).astype(np.uint8))
    else:
        mf_mask = mb_mask.copy()

    Image.fromarray(mf_mask).save(out_dir / "Mf_mask.png")
    Image.fromarray(mb_mask).save(out_dir / "Mb_mask.png")
    print(f"  Saved Mf_mask.png + Mb_mask.png → {out_dir}")
    return True


# ── Mode B: Manual (copy từ mask_output/) ─────────────────────────────────────
def copy_manual_masks(manual_mf: str, manual_mb: str, out_dir: Path, image_size: int):
    """Copy và resize mask từ path cho sẵn."""
    for src, dst_name in [(manual_mf, "Mf_mask.png"), (manual_mb, "Mb_mask.png")]:
        if not src or not Path(src).exists():
            print(f"  [WARN] Mask not found: {src} — creating empty mask")
            Image.fromarray(np.zeros((image_size, image_size), dtype=np.uint8)).save(out_dir / dst_name)
        else:
            mask = Image.open(src).convert("L").resize((image_size, image_size), Image.NEAREST)
            mask.save(out_dir / dst_name)
    print(f"  Copied manual masks → {out_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-mode", choices=["grounded-sam", "manual", "skip"],
                        default="skip",
                        help="How to generate masks")
    parser.add_argument("--manual-mf", default="mask_output/Mf_fine_grained.png",
                        help="Path to Mf mask to copy for all samples (mode=manual)")
    parser.add_argument("--manual-mb", default="mask_output/Mb_bounding_box.png",
                        help="Path to Mb mask to copy for all samples (mode=manual)")
    parser.add_argument("--samples", nargs="*", default=None,
                        help="Process only specific samples e.g. --samples sample_01 sample_03")
    args = parser.parse_args()

    target_samples = SAMPLES
    if args.samples:
        target_samples = [s for s in SAMPLES if s.name in args.samples]

    print(f"\n{'='*55}")
    print(f"  UltraEdit Eval Preprocessor")
    print(f"  Samples: {len(target_samples)} | Mask mode: {args.mask_mode}")
    print(f"{'='*55}\n")

    for sample_dir in target_samples:
        meta_path = sample_dir / "metadata.json"
        if not meta_path.exists():
            print(f"[SKIP] {sample_dir.name}: metadata.json not found")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        print(f"[{meta['sample_id']}] Processing...")

        # ── 1. Resize image ────────────────────────────────────────────────
        raw_img_path = RAW_DATA / meta["image_filename"]
        out_img_path = sample_dir / "image.png"

        if not raw_img_path.exists():
            print(f"  [WARN] Raw image not found: {raw_img_path}")
        else:
            img = Image.open(raw_img_path).convert("RGB")
            orig_size = img.size
            img_resized = resize_center_crop(img, IMAGE_SIZE)
            img_resized.save(out_img_path)
            print(f"  Image: {orig_size} → {IMAGE_SIZE}×{IMAGE_SIZE} → {out_img_path.name}")

        # ── 2. Generate masks ──────────────────────────────────────────────
        if args.mask_mode == "grounded-sam":
            if out_img_path.exists():
                ok = run_grounded_sam(out_img_path, meta["edit_region"], sample_dir)
                if not ok:
                    print(f"  [WARN] Grounded-SAM failed — masks not generated")
            else:
                print(f"  [WARN] No image to run SAM on")

        elif args.mask_mode == "manual":
            copy_manual_masks(args.manual_mf, args.manual_mb, sample_dir, IMAGE_SIZE)

        elif args.mask_mode == "skip":
            print(f"  Masks: skipped (add manually to {sample_dir}/Mf_mask.png + Mb_mask.png)")

        # ── 3. Token count report ──────────────────────────────────────────
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            for key in ["source_caption_long", "source_caption_short",
                        "target_caption_long",  "target_caption_short", "instruction"]:
                n = len(enc.encode(meta[key]))
                flag = "⚠️ >77" if n > 77 else ("⚠️ >248" if n > 248 else "")
                print(f"  {key:30s}: {n:3d} tokens {flag}")
        except ImportError:
            # tiktoken không có → dùng whitespace split ước lượng
            for key in ["source_caption_long", "source_caption_short",
                        "target_caption_long",  "target_caption_short", "instruction"]:
                n = len(meta[key].split())
                print(f"  {key:30s}: ~{n:3d} words")

        print()

    print("Preprocessing done.")
    print(f"\nNext: run `python run_eval.py` to start 5-experiment evaluation.")


if __name__ == "__main__":
    main()
