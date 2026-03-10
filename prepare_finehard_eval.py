#!/usr/bin/env python3
"""
prepare_finehard_eval.py
========================
Đọc TRY_EDA/ultraedit_samples_10.jsonl → tạo eval_data/sample_fhXX/ với:

  1. Lookup URL từ coyo parquet → download ảnh gốc
  2. Lookup full image caption từ FineHARD/json_files/ (caption + short_caption)
  3. SAM trên ảnh gốc (FULL resolution) với bbox gốc → Mf_orig
  4. Resize center-crop ảnh + mask cùng transform → image.png, Mf_mask.png
  5. Mb_mask.png  — bbox sau khi remap qua cùng crop transform
  6. Ms_mask.png  — soft fusion (Mb, Mf)
  7. metadata.json:
       source_caption_long/short = full image caption từ FineHARD JSON
       target_caption_long/short = full caption với pos_cap → neg_cap
       grounding_query           = positive_caption
       instruction               = "Change <pos> to <neg>."

⚠️  Thứ tự ĐÚNG:
  SAM(original_image, original_bbox) → Mf_orig
  crop_transform(original_image) → image.png
  crop_transform(Mf_orig)        → Mf_mask.png   ← mask khớp đúng ảnh đã crop
  remap_bbox(original_bbox, original_wh) → bbox_cropped → Mb_mask.png

Usage:
  python prepare_finehard_eval.py
  python prepare_finehard_eval.py --jsonl TRY_EDA/ultraedit_samples_10.jsonl
  python prepare_finehard_eval.py --no-sam   # skip SAM, Mf = Mb
  python prepare_finehard_eval.py --device cpu
"""

import argparse
import gc
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent
FINEHARD_DIR = REPO_ROOT / "FineHARD"
EVAL_DATA    = REPO_ROOT / "eval_data"
IMAGE_SIZE   = 512


# ══════════════════════════════════════════════════════════════════════════════
# Image utilities
# ══════════════════════════════════════════════════════════════════════════════

def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Short-side resize then center-crop to size×size."""
    w, h = img.size
    scale = size / min(w, h)
    img   = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    w, h  = img.size
    left  = (w - size) // 2
    top   = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _crop_params(orig_w: int, orig_h: int, size: int = 512):
    """Return (new_w, new_h, crop_left, crop_top) for resize+center_crop."""
    scale     = size / min(orig_w, orig_h)
    new_w     = round(orig_w * scale)
    new_h     = round(orig_h * scale)
    crop_left = (new_w - size) // 2
    crop_top  = (new_h - size) // 2
    return new_w, new_h, crop_left, crop_top


def remap_bbox_after_crop(bbox_norm: list, orig_w: int, orig_h: int,
                          size: int = 512) -> list:
    """
    Remap normalized bbox [x1,y1,x2,y2] from original image space
    to 512×512 cropped image space (same transform as resize_center_crop).
    """
    new_w, new_h, crop_left, crop_top = _crop_params(orig_w, orig_h, size)
    x1n, y1n, x2n, y2n = bbox_norm[0], bbox_norm[1], bbox_norm[2], bbox_norm[3]
    # Pixel coords in resized image
    x1 = x1n * new_w - crop_left
    y1 = y1n * new_h - crop_top
    x2 = x2n * new_w - crop_left
    y2 = y2n * new_h - crop_top
    # Clamp to [0, size]
    x1, x2 = max(0.0, min(size, x1)), max(0.0, min(size, x2))
    y1, y2 = max(0.0, min(size, y1)), max(0.0, min(size, y2))
    return [x1 / size, y1 / size, x2 / size, y2 / size]


def transform_mask_like_image(mask_np: np.ndarray,
                              orig_w: int, orig_h: int,
                              size: int = 512) -> np.ndarray:
    """
    Apply the same resize+center_crop transform to a mask (H×W uint8)
    as was applied to the source image.
    Uses NEAREST interpolation to keep hard mask values.
    """
    from PIL import Image as _Img
    new_w, new_h, crop_left, crop_top = _crop_params(orig_w, orig_h, size)
    mask_pil = _Img.fromarray(mask_np, mode="L")
    mask_pil = mask_pil.resize((new_w, new_h), _Img.NEAREST)
    mask_pil = mask_pil.crop((crop_left, crop_top, crop_left + size, crop_top + size))
    return np.array(mask_pil)


# ══════════════════════════════════════════════════════════════════════════════
# Parquet URL lookup
# ══════════════════════════════════════════════════════════════════════════════

def lookup_url(image_path: str) -> str:
    """
    image_path: "grit-20m/data-12m/coyo_image_18/00012/000123444.jpg"
    Parses → coyo_image_18/00012.parquet, key="000123444" → returns URL.
    Handles typos like "coàyo_image_15".
    """
    import pyarrow.parquet as pq

    parts = image_path.replace("\\", "/").split("/")

    # Find the coyo_image_XX segment (fix any Unicode typos)
    coyo_idx = next(
        (i for i, p in enumerate(parts) if re.match(r"coo?y?o_image_\d+", p, re.IGNORECASE)),
        None,
    )
    if coyo_idx is None:
        raise ValueError(f"Cannot parse coyo_image_XX from path: {image_path}")

    coyo_part   = re.sub(r"[^\w_]", "", parts[coyo_idx])   # strip non-ASCII
    coyo_part   = re.sub(r"^c[^o]", "co", coyo_part)        # fix "càyo" → "coyo"
    coyo_part   = "coyo_image_" + re.search(r"\d+$", coyo_part).group()

    parquet_num = parts[coyo_idx + 1]                        # e.g. "00012"
    raw_key     = parts[coyo_idx + 2]                        # e.g. "000123444.jpg"
    key         = re.sub(r"\.[^.]+$", "", raw_key)           # strip extension

    parquet_path = FINEHARD_DIR / coyo_part / f"{parquet_num}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pq.read_table(str(parquet_path), columns=["url", "key", "status"]).to_pandas()
    row = df[df["key"] == key]
    if row.empty:
        raise ValueError(f"Key '{key}' not found in {parquet_path} ({len(df):,} rows)")

    rec    = row.iloc[0]
    status = rec["status"]
    url    = rec["url"]
    print(f"  URL [{status}]: {url[:90]}...")
    if status != "success":
        raise ValueError(f"Image download status was '{status}' in dataset — may be unavailable")

    return url


def download_image(url: str, save_path: Path, timeout: int = 20):
    """HTTP GET with browser-like UA header."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "image/*,*/*",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    save_path.write_bytes(data)


# ══════════════════════════════════════════════════════════════════════════════
# FineHARD JSON caption lookup
# ══════════════════════════════════════════════════════════════════════════════

def lookup_finehard_caption(sample_id: str) -> dict:
    """
    Tìm full image caption từ FineHARD/json_files/ dựa trên sample_id.

    sample_id format: "18_123444_0_0"  →  image_id = "18_123444", bbox_idx = 0
    Dùng grep để tìm nhanh file JSON chứa image_id đó.

    Returns dict với:
      caption       : full long caption (str)
      short_caption : short caption (str)
      short_expr    : bbox_info[bbox_idx]["short_expr"] (str)
    hoặc {} nếu không tìm thấy.
    """
    import subprocess

    # Extract image_id = first two parts of sample_id (split_coyo + image_key)
    # Format: "{split}_{imagekey}_{bbox_idx}_{neg_idx}"
    parts    = sample_id.split("_")
    img_id   = "_".join(parts[:2])   # e.g. "18_123444"
    bbox_idx = int(parts[2]) if len(parts) > 2 else 0

    json_dir = FINEHARD_DIR / "json_files"
    if not json_dir.exists():
        print(f"  ⚠️  json_files/ not found, skipping caption lookup")
        return {}

    # grep -l searches file list; --include for *.json
    print(f"  🔍 Searching FineHARD JSON for image_id={img_id} ...")
    try:
        result = subprocess.run(
            ["grep", "-rl", f'"id": "{img_id}"', str(json_dir)],
            capture_output=True, text=True, timeout=30,
        )
        matches = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    except subprocess.TimeoutExpired:
        print("  ⚠️  grep timeout — caption lookup skipped")
        return {}

    if not matches:
        print(f"  ⚠️  image_id={img_id} not found in json_files/")
        return {}

    json_path = Path(matches[0])
    print(f"  📄 Found in {json_path.name}")

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        if rec.get("id") == img_id:
            cap        = rec.get("caption", "")
            short      = rec.get("short_caption", "")
            bbox_infos = rec.get("bbox_info", [])
            short_expr = ""
            if bbox_idx < len(bbox_infos):
                short_expr = bbox_infos[bbox_idx].get("short_expr", "")
            print(f"  ✅ Caption ({len(cap.split())} words): {cap[:80]}...")
            print(f"  ✅ short_expr [{bbox_idx}]: {short_expr}")
            return {"caption": cap, "short_caption": short, "short_expr": short_expr}

    print(f"  ⚠️  image_id={img_id} found by grep but not matched in records")
    return {}


def build_short_caption_pair(short_cap: str, pos_cap: str, neg_cap: str,
                             max_words: int = 35) -> tuple:
    """
    Build (source_short, target_short) ≤ max_words each, using FineHARD
    short_caption as base — NO long caption involved.

    Rules:
    - If short_cap already contains pos_cap → trim to max_words, use as-is.
    - Otherwise trim short_cap to (max_words - len(pos_cap) - 2) words,
      then append ", featuring {pos_cap}".
    - Target = replace pos_cap → neg_cap.
    """
    has_pos = bool(re.search(re.escape(pos_cap), short_cap, re.IGNORECASE))
    words   = short_cap.split()

    if has_pos:
        src = " ".join(words[:max_words])
    else:
        budget = max(1, max_words - len(pos_cap.split()) - 2)
        base   = " ".join(words[:budget]).rstrip(".,;")
        src    = f"{base}, featuring {pos_cap}"

    tgt = re.sub(re.escape(pos_cap), neg_cap, src, count=1, flags=re.IGNORECASE)
    return src, tgt


def build_target_caption(src_caption: str, pos_cap: str, neg_cap: str) -> str:
    """
    Replace first occurrence of pos_cap (case-insensitive) with neg_cap.
    If not found verbatim, returns src_caption unchanged.
    """
    pattern = re.compile(re.escape(pos_cap), re.IGNORECASE)
    result, n = pattern.subn(neg_cap, src_caption, count=1)
    if n == 0:
        return src_caption   # better to leave unchanged than append gibberish
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Mask utilities
# ══════════════════════════════════════════════════════════════════════════════

def make_mb_mask(bbox_norm: list, size: int = 512) -> np.ndarray:
    """
    bbox_norm = [x1, y1, x2, y2, (optional conf_score)]  — all in [0, 1].
    Returns Mb: HxW uint8, 255 inside bbox, 0 outside.
    """
    x1n, y1n = bbox_norm[0], bbox_norm[1]
    x2n, y2n = bbox_norm[2], bbox_norm[3]
    x1, y1   = max(0, int(x1n * size)), max(0, int(y1n * size))
    x2, y2   = min(size, int(x2n * size)), min(size, int(y2n * size))
    Mb = np.zeros((size, size), dtype=np.uint8)
    Mb[y1:y2, x1:x2] = 255
    return Mb


def run_sam_with_bbox(image_np: np.ndarray, bbox_norm: list, device: str) -> np.ndarray:
    """
    SAM segmentation with a box prompt from the dataset's normalized bbox.
    image_np  = ORIGINAL full-resolution image (before any resize/crop).
    bbox_norm = [x1, y1, x2, y2] in original normalized coords (0-1).
    Returns Mf (H, W) uint8 at ORIGINAL resolution.
    Caller must apply transform_mask_like_image() afterward.
    """
    import torch
    from torchvision.ops import box_convert
    from segment_anything import build_sam, SamPredictor

    gsa_dir  = REPO_ROOT / "data_generation" / "Grounded-Segment-Anything"
    sam_ckpt = gsa_dir / "model" / "sam" / "sam_vit_h_4b8939.pth"

    if not sam_ckpt.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}")

    # Ensure GSA path is importable
    for p in [str(REPO_ROOT), str(gsa_dir), str(REPO_ROOT / "data_generation")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    sam           = build_sam(checkpoint=str(sam_ckpt)).to(device)
    sam_predictor = SamPredictor(sam)

    orig_h, orig_w = image_np.shape[:2]

    # Pre-resize to max 1024px longest side before set_image (CPU speed + memory).
    # bbox_norm is scale-invariant, so no coord adjustment needed.
    MAX_SAM_SIDE = 1024
    scale_sam = min(1.0, MAX_SAM_SIDE / max(orig_w, orig_h))
    if scale_sam < 1.0:
        from PIL import Image as _PILImg
        sam_w = round(orig_w * scale_sam)
        sam_h = round(orig_h * scale_sam)
        img_for_sam = np.array(
            _PILImg.fromarray(image_np).resize((sam_w, sam_h), _PILImg.LANCZOS)
        )
        print(f"  [SAM] Pre-resize {orig_w}×{orig_h} → {sam_w}×{sam_h} for speed")
    else:
        img_for_sam = image_np
        sam_w, sam_h = orig_w, orig_h

    sam_predictor.set_image(img_for_sam)

    h, w = img_for_sam.shape[:2]
    x1n, y1n, x2n, y2n = bbox_norm[0], bbox_norm[1], bbox_norm[2], bbox_norm[3]

    # Dataset bbox is xyxy-normalized → convert to cxcywh-normalized
    # (matches GroundingDINO output format expected by SAM helper)
    cx   = (x1n + x2n) / 2
    cy   = (y1n + y2n) / 2
    bw   = x2n - x1n
    bh   = y2n - y1n
    boxes_cxcywh = torch.tensor([[cx, cy, bw, bh]], dtype=torch.float32)

    # Scale to pixel coords then apply SAM transform
    pixel_boxes = boxes_cxcywh * torch.tensor([w, h, w, h], dtype=torch.float32)
    xyxy = box_convert(boxes=pixel_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    transformed = sam_predictor.transform.apply_boxes_torch(xyxy.to(device), (h, w))

    with torch.no_grad():
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )

    merged = torch.max(masks.cpu(), 0)[0][0].numpy()
    Mf_sam = (merged * 255).astype(np.uint8)
    print(f"  [SAM] Mf coverage (sam-res): {(Mf_sam > 0).mean() * 100:.1f}%  "
          f"bbox area: {(x2n-x1n)*(y2n-y1n)*100:.1f}%")

    # Scale mask back to original resolution if we pre-resized
    if scale_sam < 1.0:
        from PIL import Image as _PILImg
        Mf = np.array(
            _PILImg.fromarray(Mf_sam).resize((orig_w, orig_h), _PILImg.NEAREST)
        )
    else:
        Mf = Mf_sam

    # Clean up VRAM
    del sam, sam_predictor, masks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Mf


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    def _cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    parser = argparse.ArgumentParser(
        description="Prepare FineHARD JSONL samples as eval_data/sample_fhXX/ for modal_2stage"
    )
    parser.add_argument(
        "--jsonl", default="TRY_EDA/ultraedit_samples_10.jsonl",
        help="Path to JSONL file (relative to repo root)",
    )
    parser.add_argument(
        "--no-sam", action="store_true",
        help="Skip SAM segmentation; use bbox mask (Mb) as both Mb and Mf",
    )
    parser.add_argument(
        "--device", default="cuda" if _cuda_available() else "cpu",
        help="Torch device for SAM (default: cuda if available)",
    )
    parser.add_argument(
        "--soft-mask-value", type=float, default=0.5,
        help="Soft mask blend factor for region Mb\\Mf (default 0.5)",
    )
    parser.add_argument(
        "--prefix", default="fh",
        help="Prefix for sample dirs: eval_data/sample_<prefix>XX (default: fh)",
    )
    parser.add_argument(
        "--start-num", type=int, default=None,
        help="Override starting sample number (default: auto-detect from existing dirs)",
    )
    args = parser.parse_args()

    # ── Load JSONL ──────────────────────────────────────────────────────────
    jsonl_path = REPO_ROOT / args.jsonl
    if not jsonl_path.exists():
        sys.exit(f"❌ JSONL not found: {jsonl_path}")

    entries = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"📋 Loaded {len(entries)} entries from {jsonl_path.name}")

    # ── Import soft mask helper ─────────────────────────────────────────────
    sys.path.insert(0, str(REPO_ROOT))
    from create_soft_mask import step5_soft_mask_fusion

    EVAL_DATA.mkdir(exist_ok=True)

    # ── Find starting sample number ─────────────────────────────────────────
    existing_nums = [
        int(d.name[len(f"sample_{args.prefix}"):])
        for d in EVAL_DATA.glob(f"sample_{args.prefix}*/")
        if d.name[len(f"sample_{args.prefix}"):].isdigit()
    ]
    next_num = args.start_num if args.start_num is not None else (
        max(existing_nums, default=0) + 1
    )

    created = []

    for entry in entries:
        sid     = entry["sample_id"]
        img_rel = entry["image_path"]
        bbox    = entry["bbox"]           # [x1,y1,x2,y2] or [x1,y1,x2,y2,conf]
        pos_cap = entry["positive_caption"]
        neg_cap = entry["hard_negative_caption"]

        sample_dir = EVAL_DATA / f"sample_{args.prefix}{next_num:02d}"
        sample_dir.mkdir(exist_ok=True)
        next_num  += 1

        print(f"\n{'─'*60}")
        print(f"[{sid}]  →  {sample_dir.name}")
        print(f"  BBox (norm) : {[round(b, 4) for b in bbox[:4]]}")
        print(f"  Positive    : {pos_cap}")
        print(f"  Negative    : {neg_cap}")

        # ── Step 1: Download image ──────────────────────────────────────────
        raw_img_path = sample_dir / "raw_source.jpg"
        if not raw_img_path.exists():
            print("  🔗 Looking up URL in parquet...")
            try:
                url = lookup_url(img_rel)
                print("  ⬇️  Downloading...")
                download_image(url, raw_img_path)
                size_kb = raw_img_path.stat().st_size / 1024
                print(f"  ✅ Downloaded ({size_kb:.0f} KB)")
            except Exception as e:
                print(f"  ❌ Download failed: {e}")
                print(f"     Skipping {sample_dir.name}")
                sample_dir.rmdir()
                next_num -= 1
                continue
        else:
            print(f"  ✅ Image cached: {raw_img_path.name}")

        # ── Step 2: Load original image (keep full-res for SAM) ──────────────
        try:
            pil_src = Image.open(raw_img_path).convert("RGB")
            orig_w, orig_h = pil_src.size
        except Exception as e:
            print(f"  ❌ Cannot open image: {e}")
            continue

        image_np_orig = np.array(pil_src)   # full-res for SAM

        # ── Step 3a: SAM on ORIGINAL full-res image → Mf_orig ────────────────
        # ⚠️  Must run BEFORE crop so bbox coords map correctly
        Mf_orig = None
        if not args.no_sam:
            print(f"  🔄 Running SAM on original {orig_w}×{orig_h} (device={args.device})...")
            try:
                Mf_orig = run_sam_with_bbox(image_np_orig, bbox, args.device)
            except FileNotFoundError as e:
                print(f"  ⚠️  {e}")
            except Exception as e:
                print(f"  ⚠️  SAM error: {e}")

        # ── Step 3b: Apply resize+crop to image ──────────────────────────────
        img_png_path = sample_dir / "image.png"
        pil_512 = resize_center_crop(pil_src, IMAGE_SIZE)
        pil_512.save(img_png_path)
        print(f"  ✅ Resized {orig_w}×{orig_h} → {IMAGE_SIZE}×{IMAGE_SIZE} → image.png")

        # ── Step 3c: Apply SAME transform to Mf_orig → Mf (512×512) ─────────
        if Mf_orig is not None:
            Mf = transform_mask_like_image(Mf_orig, orig_w, orig_h, IMAGE_SIZE)
            print(f"  ✅ SAM mask transformed: {(Mf > 0).mean()*100:.1f}% coverage @ 512×512")
        else:
            Mf = None
            print(f"  ℹ️  SAM unavailable — will use Mb as Mf")

        # ── Step 4: Mb mask — bbox remapped through same crop transform ───────
        bbox_cropped = remap_bbox_after_crop(bbox, orig_w, orig_h, IMAGE_SIZE)
        Mb     = make_mb_mask(bbox_cropped, IMAGE_SIZE)
        mb_pct = (Mb > 0).mean() * 100
        print(f"  ✅ Mb mask: {mb_pct:.1f}% coverage "
              f"(bbox_orig {bbox[0]:.3f},{bbox[1]:.3f}→{bbox[2]:.3f},{bbox[3]:.3f} "
              f"→ cropped {bbox_cropped[0]:.3f},{bbox_cropped[1]:.3f}"
              f"→{bbox_cropped[2]:.3f},{bbox_cropped[3]:.3f})")

        if Mf is None:
            Mf = Mb.copy()
            if args.no_sam:
                print(f"  ℹ️  --no-sam: using Mb as Mf")

        # ── Step 5: Ms = soft fusion ─────────────────────────────────────────
        Ms = step5_soft_mask_fusion(Mb, Mf, args.soft_mask_value)

        # ── Step 6: Save masks ────────────────────────────────────────────────
        Image.fromarray(Mb).save(sample_dir / "Mb_mask.png")
        Image.fromarray(Mf).save(sample_dir / "Mf_mask.png")
        Image.fromarray(Ms).save(sample_dir / "Ms_mask.png")
        print(f"  ✅ Saved Mb_mask.png / Mf_mask.png / Ms_mask.png")

        # ── Step 7: Captions from FineHARD bbox_info ──────────────────────────
        # query (SAM segment) = short_expr từ FineHARD bbox_info trực tiếp
        # source_caption      = positive_caption (long_expr) — mô tả region hiện tại
        # target_caption      = hard_negative_caption — chỉ thay phần cần thay
        fh_caps    = lookup_finehard_caption(sid)
        short_expr = fh_caps.get("short_expr", "") if fh_caps else ""
        if not short_expr:
            short_expr = pos_cap
            print(f"  ⚠️  No short_expr found, using pos_cap as grounding_query")

        src_cap = pos_cap   # = long_expr: full positive description of region
        # target_long = pos_cap với short_expr được thay bằng neg_cap
        # e.g. "a low wooden table with carved design..." → "a copper table with carved design..."
        tgt_cap_long  = build_target_caption(pos_cap, short_expr, neg_cap)
        tgt_cap_short = neg_cap   # short form = just the hard negative
        print(f"  📝 grounding_query: {short_expr}")
        print(f"  📝 src ({len(src_cap.split())}w): {src_cap}")
        print(f"  📝 tgt_long ({len(tgt_cap_long.split())}w): {tgt_cap_long}")
        print(f"  📝 tgt_short: {tgt_cap_short}")

        # ── Step 8: metadata.json ─────────────────────────────────────────────
        # grounding_query = short_expr → SAM query trực tiếp, KHÔNG dựa vào instruction
        # source_caption  = positive_caption (long_expr)
        # target_caption  = hard_negative_caption
        metadata = {
            "sample_id":            sample_dir.name,
            "finehard_id":          sid,
            "image_filename":       "",
            "grounding_query":      short_expr,
            "instruction":          f'Change "{short_expr}" to "{neg_cap}".',
            "edit_region":          short_expr,
            "source_caption_short": src_cap,
            "source_caption_long":  src_cap,
            "target_caption_short": tgt_cap_short,
            "target_caption_long":  tgt_cap_long,
            "bbox_norm":            [round(b, 6) for b in bbox_cropped],
            "bbox_norm_original":   [round(b, 6) for b in bbox[:4]],
            "original_image_path":  img_rel,
        }
        meta_path = sample_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✅ metadata.json saved")

        created.append(sample_dir.name)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ Created {len(created)} eval sample(s):\n")
    for name in created:
        print(f"   eval_data/{name}/")

    if not created:
        print("(none created — check errors above)")
        return

    sample_list_space = " ".join(created)
    sample_list_comma = ",".join(created)

    print(f"""
{'─'*60}
📌 Bước tiếp theo:

▶ Option A — dùng SAM masks đã sinh ở bước này (nhanh):
  python run_eval_preprocess.py --mask-mode skip \\
    --samples {sample_list_space}

▶ Option B — regenerate masks với Grounded-SAM (query = positive_caption):
  python run_eval_preprocess.py --mask-mode grounded-sam --force-mask \\
    --samples {sample_list_space}

▶ Chạy Modal 2-stage (sau khi có image.png + masks):
  modal run modal_2stage_100.py::run_all \\
    --samples "{sample_list_comma}"

▶ Chạy song song tất cả encoder:
  modal run modal_2stage_100.py::run_all \\
    --longclip-encoder both \\
    --samples "{sample_list_comma}"
{'─'*60}""")


if __name__ == "__main__":
    main()
