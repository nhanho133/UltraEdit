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


# ── Step 1a: RAM (Recognize-Anything) → Object Tags ──────────────────────────
def step1_ram_detect(pil_image: Image.Image, gsa_dir: Path, device: str) -> list:
    """
    Paper Step 1 (first half): Run RAM to get all detected object/attribute tags.
    Returns list of tag strings, or [] if RAM weights not available.
    """
    import sys

    # Search common checkpoint locations
    ram_ckpt = None
    for candidate in [
        gsa_dir / "model" / "ram" / "ram_swin_large_14m.pth",
        gsa_dir / "ram_swin_large_14m.pth",
        gsa_dir / "model" / "ram_swin_large_14m.pth",
    ]:
        if candidate.exists():
            ram_ckpt = candidate
            break

    if ram_ckpt is None:
        return []

    try:
        import torch
        if str(gsa_dir) not in sys.path:
            sys.path.insert(0, str(gsa_dir))
        from ram.models import ram as build_ram
        from ram import inference_ram, get_transform

        transform   = get_transform(image_size=384)
        img_tensor  = transform(pil_image).unsqueeze(0).to(device)
        model       = build_ram(pretrained=str(ram_ckpt), image_size=384, vit="swin_l")
        model.eval().to(device)
        with torch.no_grad():
            result = inference_ram(img_tensor, model)
        tags = [t.strip() for t in result[0].split("|") if t.strip()]
        # Free RAM model from GPU before GroundingDINO loads
        del model, img_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return tags
    except Exception as e:
        print(f"  [Step1-RAM] Not available ({type(e).__name__}), skipping")
        return []


# Paper's Table 10 prompt (Appendix A.3)
_LLM_PROMPT = """\
The following provides an instruction for image editing, an original image caption, \
a revised caption after the edit, and a set of objects detected in the image.

Edit Instruction: "{instruction}"
Original Caption: "{source_caption}"
Revised Caption:  "{target_caption}"
Detected Objects: [{object_list}]

Your task: Identify the object(s) most likely to be modified by this instruction.

Rules:
1. If the instruction modifies the ENTIRE image (style transfer, season/lighting/weather change), \
respond with exactly: GLOBAL
2. Otherwise respond with 1-2 object names from the detected list, comma-separated.
3. Use the shortest searchable form (e.g. "dog" not "brown fluffy dog").
4. No extra explanation.

Response:"""


# ── Step 1b: LLM → Target Object Keyword ──────────────────────────────────────
def step1_llm_target(instruction: str, source_caption: str, target_caption: str,
                     object_list: list) -> tuple:
    """
    Paper Step 1 (second half): identify which object to edit.
    Priority: OpenAI API (gpt-4o-mini) → simple heuristic.
    Returns (keyword: str, is_global: bool).
    """
    import os, re

    obj_str = ", ".join(f'"{o}"' for o in object_list) if object_list else "(none detected)"

    # ── Try OpenAI ─────────────────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = _LLM_PROMPT.format(
                instruction=instruction,
                source_caption=source_caption,
                target_caption=target_caption,
                object_list=obj_str,
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.0,
            )
            answer = resp.choices[0].message.content.strip()
            print(f"  [Step1-LLM] OpenAI response: \"{answer}\"")
            if answer.upper().startswith("GLOBAL"):
                return ("", True)
            keyword = answer.split(",")[0].strip().strip('"').lower()
            return (keyword, False)
        except Exception as e:
            print(f"  [Step1-LLM] OpenAI failed ({e}), using heuristic")

    # ── Heuristic fallback ─────────────────────────────────────────────────────
    instr_lower = instruction.lower()

    global_kws = ["style", "oil paint", "watercolor", "sketch", "cartoon",
                  "season", "weather", "lighting", "entire image", "whole image"]
    if any(kw in instr_lower for kw in global_kws):
        return ("", True)

    # Match RAM tags directly against instruction text
    for obj in object_list:
        if obj.lower() in instr_lower:
            return (obj.lower(), False)

    # ── Extract quoted source object: 'change "X" to "Y"' → X
    # Handles: Change "CHINA" to "USA",  replace "cat" with "dog", etc.
    m_quoted = re.search(
        r'(?:add|remove|change|replace|turn|rename|relabel|rewrite|swap)'
        r'[^"]*"([^"]+)"',
        instruction,  # keep original casing for the captured group
        re.IGNORECASE,
    )
    if m_quoted:
        return (m_quoted.group(1).strip(), False)

    # ── Extract noun phrase after action verb (original heuristic) ────────────
    m = re.search(
        r'(?:add|remove|change|replace|turn|make|move|put|place|delete|insert)'
        r'\s+(?:a|an|the)?\s*([a-z][a-z\s]{1,25?})'
        r'(?:\s+(?:to|from|into|in|on|at|with)|$)',
        instr_lower,
    )
    if m:
        return (m.group(1).strip(), False)

    return ("", False)


# ── Step 1: Full RAM + LLM → grounding_query ──────────────────────────────────
def run_step1_query(pil_image: Image.Image, meta: dict,
                    gsa_dir: Path, device: str) -> str:
    """
    Paper Step 1: RAM → LLM → grounding_query.

    Priority cascade:
      1. meta['grounding_query']  — explicit per-sample override (highest priority)
      2. RAM + LLM               — paper's automatic method
      3. meta['edit_region']     — last fallback (long phrase, less accurate)

    Returns the query string to pass to GroundingDINO, or "" for global edit
    (caller should create a full-image mask in that case).
    """
    # Level 1: explicit override
    if meta.get("grounding_query"):
        print(f"  [Step1] override from metadata: \"{meta['grounding_query']}\"")
        return meta["grounding_query"]

    print("  [Step1] No grounding_query override — running RAM → LLM...")

    # Level 2a: RAM
    object_list = step1_ram_detect(pil_image, gsa_dir, device)
    if object_list:
        print(f"  [Step1-RAM] {len(object_list)} tags: {object_list[:8]}{'...' if len(object_list)>8 else ''}")
    else:
        print("  [Step1-RAM] Unavailable — LLM will work without object list")

    # Level 2b: LLM
    src_cap  = meta.get("source_caption_short") or meta.get("source_caption_long", "")
    tgt_cap  = meta.get("target_caption_short") or meta.get("target_caption_long", "")
    keyword, is_global = step1_llm_target(
        meta.get("instruction", ""), src_cap, tgt_cap, object_list
    )

    if is_global:
        print("  [Step1] GLOBAL _— full-image mask will be used")
        return ""

    if keyword:
        print(f"  [Step1-LLM] Target object: \"{keyword}\"")
        return keyword

    # Level 3: fallback
    fallback = meta.get("edit_region", "")
    print(f"  [Step1] Fallback to edit_region: \"{fallback[:60]}\"")
    return fallback


# ── Mode A: Grounded-SAM ───────────────────────────────────────────────────────
def run_grounded_sam(image_path: Path, text_query: str, out_dir: Path,
                     soft_mask_value: float = 0.5):
    """
    Dùng Grounded-SAM để sinh Mb + Mf + Ms từ text_query.
    text_query = "" → global edit → full-image mask (Mf = Mb = Ms = all 255).
    """
    import sys
    import torch

    # Global edit → full-image mask
    if not text_query:
        print("  [Grounded-SAM] Global edit: creating full-image mask")
        full = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)
        Image.fromarray(full).save(out_dir / "Mf_mask.png")
        Image.fromarray(full).save(out_dir / "Mb_mask.png")
        Image.fromarray(full).save(out_dir / "Ms_mask.png")
        return True

    # Setup sys.path cho GroundingDINO + SAM
    gsa_dir = REPO_ROOT / "data_generation" / "Grounded-Segment-Anything"
    for p in [str(REPO_ROOT), str(gsa_dir), str(REPO_ROOT / "data_generation")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Import các hàm ổn định từ create_soft_mask.py (single source of truth)
    from create_soft_mask import (
        load_groundingdino_model,
        step3_grounding_dino,
        step4_sam,
        step5_soft_mask_fusion,
    )
    from segment_anything import build_sam, SamPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dino_cfg  = gsa_dir / "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
    dino_ckpt = gsa_dir / "model/grounding_dino/groundingdino_swinb_cogcoor.pth"
    sam_ckpt  = gsa_dir / "model/sam/sam_vit_h_4b8939.pth"

    missing = [p for p in [dino_ckpt, sam_ckpt] if not p.exists()]
    if missing:
        print(f"  [WARN] Grounded-SAM weights not found:")
        for p in missing:
            print(f"    {p}")
        print(f"  [WARN] Skipping mask generation for {image_path.name}")
        return False

    pil_image = Image.open(image_path).convert("RGB")

    # ── Bước 3: GroundingDINO → Mb (có fallback threshold tự động) ────────────
    gd_model = load_groundingdino_model(str(dino_cfg), str(dino_ckpt), device)
    Mb, boxes, image_np = step3_grounding_dino(
        pil_image, text_query, gd_model, device)
    del gd_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Bước 4: SAM → Mf ──────────────────────────────────────────────────────
    sam = build_sam(checkpoint=str(sam_ckpt)).to(device)
    sam_predictor = SamPredictor(sam)
    Mf = step4_sam(image_np, boxes, sam_predictor, device)
    del sam, sam_predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Bước 5: Soft mask fusion → Ms ─────────────────────────────────────────
    Ms = step5_soft_mask_fusion(Mb, Mf, soft_mask_value)

    Image.fromarray(Mf).save(out_dir / "Mf_mask.png")
    Image.fromarray(Mb).save(out_dir / "Mb_mask.png")
    Image.fromarray(Ms).save(out_dir / "Ms_mask.png")
    print(f"  Saved Mf_mask.png + Mb_mask.png + Ms_mask.png → {out_dir}")
    return True


# ── Post-processing: geometric adjustments to saved masks ─────────────────────
def apply_mask_post_process(pp: dict, out_dir: Path, soft_mask_value: float = 0.5):
    """
    Apply a geometric post-process to Mf/Mb/Ms masks already saved on disk.
    Supported ops:
      crop_x   : zero out pixels where x >= x_max_fraction * W
                 (keeps a left portion; use when target region is on the left)
      expand_top: extend the mask upward by expand_top_px pixels
                 (use when detected region is too small for a large object addition)
    After modifying Mf and Mb, recomputes Ms (soft fusion) from scratch.
    """
    from create_soft_mask import step5_soft_mask_fusion

    mf = np.array(Image.open(out_dir / "Mf_mask.png").convert("L"))
    mb = np.array(Image.open(out_dir / "Mb_mask.png").convert("L"))
    H, W = mf.shape
    op = pp.get("op")

    if op == "crop_x":
        x_max = int(pp["x_max_fraction"] * W)
        mf[:, x_max:] = 0
        mb[:, x_max:] = 0
        print(f"  [post_process] crop_x: zeroed x>{x_max} (kept left {pp['x_max_fraction']*100:.0f}%)")

    elif op == "crop_bbox":
        # Keep only pixels inside a fractional bounding box [x_min,y_min,x_max,y_max]
        # Useful when the target region (e.g. cargo bed interior) is a precise sub-area.
        x0 = int(pp.get("x_min_fraction", 0.0) * W)
        x1 = int(pp.get("x_max_fraction", 1.0) * W)
        y0 = int(pp.get("y_min_fraction", 0.0) * H)
        y1 = int(pp.get("y_max_fraction", 1.0) * H)
        # Zero everything outside the bbox
        out_mask = np.zeros_like(mf)
        out_mask[y0:y1, x0:x1] = 1
        mf = (mf.astype(float) * out_mask).astype(np.uint8)
        mb = (mb.astype(float) * out_mask).astype(np.uint8)
        print(f"  [post_process] crop_bbox: kept x={x0}-{x1}, y={y0}-{y1} "
              f"({(x1-x0)/W*100:.0f}%W × {(y1-y0)/H*100:.0f}%H)")

    elif op == "expand_top":
        expand_px = int(pp["expand_top_px"])
        # Find the top row of the current mask
        rows_with_mask = np.where(np.any(mf > 127, axis=1))[0]
        if len(rows_with_mask) == 0:
            print(f"  [post_process] expand_top: no mask pixels found, skipping")
            return
        top_row = int(rows_with_mask[0])
        new_top = max(0, top_row - expand_px)
        # Expand both Mf and Mb upward as a full-width band
        mf[new_top:top_row, :] = 255
        mb[new_top:top_row, :] = 255
        print(f"  [post_process] expand_top: extended mask from y={top_row} → y={new_top} (+{expand_px}px)")

    else:
        print(f"  [WARN] Unknown post_process op: {op!r} — skipping")
        return

    # Optional: promote Mb -> Mf so the entire cropped region gets value 1.0
    # (use when SAM doesn't segment well inside the target area, e.g. open spaces)
    if pp.get("promote_mb_to_mf", False):
        mf = mb.copy()
        print(f"  [post_process] promote_mb_to_mf: Mf set equal to Mb")

    # Recompute Ms from modified Mf + Mb
    Ms = step5_soft_mask_fusion(mb, mf, soft_mask_value)
    Image.fromarray(mf).save(out_dir / "Mf_mask.png")
    Image.fromarray(mb).save(out_dir / "Mb_mask.png")
    Image.fromarray(Ms).save(out_dir / "Ms_mask.png")

    mf_cov = (mf > 127).mean() * 100
    mb_cov = (mb > 127).mean() * 100
    print(f"  [post_process] After: Mf={mf_cov:.1f}%, Mb={mb_cov:.1f}%")


# ── Mode C: OCR Polygon Mask (PaddleOCR) ─────────────────────────────────────
def run_ocr_mask(image_path: Path, search_word: str, out_dir: Path,
                soft_mask_value: float = 0.5, expand_px: int = 6):
    """
    Dùng EasyOCR detect text boxes → tìm word khớp search_word → polygon mask.
    Mf = polygon fill của word box
    Mb = bounding rect mở rộng của Mf
    Ms = soft fusion(Mb, Mf)
    """
    try:
        import easyocr
    except ImportError:
        print("  [OCR] EasyOCR chưa cài: pip install easyocr")
        return False

    import cv2
    import os

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"  [OCR] Cannot read image: {image_path}")
        return False
    h, w = img_bgr.shape[:2]

    print(f"  [OCR] Running EasyOCR, searching for: \"{search_word}\"")
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(REPO_ROOT / ".easyocr"))
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(img_bgr)
    # results: list of (bbox [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], text, confidence)

    target = search_word.strip().upper()
    Mf     = np.zeros((h, w), dtype=np.uint8)
    found  = []

    for (box, text, score) in results:
        if target in text.upper():
            pts = np.array(box).astype(np.int32)
            # Expand polygon outward by expand_px
            if expand_px > 0:
                cx, cy    = pts.mean(axis=0)
                direction = pts - np.array([cx, cy])
                norm      = np.linalg.norm(direction, axis=1, keepdims=True)
                norm      = np.where(norm < 1e-6, 1, norm)
                pts       = (pts + (direction / norm * expand_px)).astype(np.int32)
            cv2.fillPoly(Mf, [pts], 255)
            found.append((text, score))
            print(f"  [OCR] Found: \"{text}\" (score={score:.2f})")

    if not found:
        print(f"  [OCR] WARNING: \"{search_word}\" not found in OCR results")
        all_texts = [text for (_, text, _) in results]
        print(f"  [OCR] All detected words: {all_texts}")
        return False

    # Mb = bounding createrect của toàn bộ Mf pixels + padding
    Mb   = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.where(Mf > 0)
    if len(xs) > 0:
        pad = 10
        x1 = max(0, xs.min() - pad)
        y1 = max(0, ys.min() - pad)
        x2 = min(w, xs.max() + pad)
        y2 = min(h, ys.max() + pad)
        Mb[y1:y2, x1:x2] = 255
        print(f"  [OCR] Mf coverage: {(Mf>0).mean()*100:.1f}%  "
              f"Mb bbox: ({x1},{y1})-({x2},{y2})")

    from create_soft_mask import step5_soft_mask_fusion
    Ms = step5_soft_mask_fusion(Mb, Mf, soft_mask_value)

    Image.fromarray(Mf).save(out_dir / "Mf_mask.png")
    Image.fromarray(Mb).save(out_dir / "Mb_mask.png")
    Image.fromarray(Ms).save(out_dir / "Ms_mask.png")
    print(f"  Saved Mf_mask.png + Mb_mask.png + Ms_mask.png → {out_dir}")
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
    parser.add_argument("--mask-mode", "--maskmode", dest="mask_mode",
                        choices=["grounded-sam", "ocr", "manual", "skip"],
                        default="skip",
                        help="How to generate masks. "
                             "'ocr' = PaddleOCR polygon mask (dùng cho text-edit tasks)")
    parser.add_argument("--manual-mf", default="mask_output/Mf_fine_grained.png",
                        help="Path to Mf mask to copy for all samples (mode=manual)")
    parser.add_argument("--manual-mb", default="mask_output/Mb_bounding_box.png",
                        help="Path to Mb mask to copy for all samples (mode=manual)")
    parser.add_argument("--samples", nargs="*", default=None,
                        help="Process only specific samples e.g. --samples sample_01 sample_03")
    parser.add_argument("--query", default=None,
                        help="Override grounding_query cho tất cả samples được chỉ định, "
                             "bỏ qua RAM/LLM và metadata. "
                             "VD: --query \"graffiti CHINA\"")
    parser.add_argument("--soft-mask-value", type=float, default=0.5,
                        help="s ∈ [0.0,1.0] — blend factor vùng Mb\\Mf trong soft mask (mặc định 0.5)")
    parser.add_argument("--force-mask", action="store_true",
                        help="Overwrite existing Mf/Mb/Ms masks even if they already exist")
    args = parser.parse_args()

    target_samples = SAMPLES
    if args.samples:
        target_samples = [s for s in SAMPLES if s.name in args.samples]

    print(f"\n{'='*55}")
    print(f"  UltraEdit Eval Preprocessor")
    print(f"  Samples: {len(target_samples)} | Mask mode: {args.mask_mode}")
    if args.query:
        print(f"  Query override: \"{args.query}\"")
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
        img_filename = meta.get("image_filename", "").strip()
        out_img_path = sample_dir / "image.png"

        if not img_filename:
            # No raw source → check if image.png or 1.png already in sample dir
            if out_img_path.exists():
                print(f"  Image: using existing image.png (no image_filename in metadata)")
            else:
                local_fallback = next(
                    (sample_dir / f for f in ["1.png", "2.png", "source.png"]
                     if (sample_dir / f).exists()), None
                )
                if local_fallback:
                    img = Image.open(local_fallback).convert("RGB")
                    img_resized = resize_center_crop(img, IMAGE_SIZE)
                    img_resized.save(out_img_path)
                    print(f"  Image: {local_fallback.name} → {IMAGE_SIZE}×{IMAGE_SIZE} → image.png")
                else:
                    print(f"  [WARN] No image source found for {sample_dir.name} — skipping")
        else:
            raw_img_path = RAW_DATA / img_filename
            if not raw_img_path.exists():
                print(f"  [WARN] Raw image not found: {raw_img_path}")
            else:
                img = Image.open(raw_img_path).convert("RGB")
                orig_size = img.size
                img_resized = resize_center_crop(img, IMAGE_SIZE)
                img_resized.save(out_img_path)
                print(f"  Image: {orig_size} → {IMAGE_SIZE}×{IMAGE_SIZE} → {out_img_path.name}")

        # ── 2. Generate masks ──────────────────────────────────────────────
        masks_exist = all((sample_dir / f).exists() for f in ["Mf_mask.png", "Mb_mask.png"])
        if masks_exist and not args.force_mask and args.mask_mode != "skip":
            print(f"  Masks: already exist (use --force-mask to regenerate)")
        elif args.mask_mode == "grounded-sam":
            if out_img_path.exists():
                # Paper Step 1: RAM → LLM → grounding_query
                # Priority: --query CLI arg → meta['grounding_query'] override → RAM+LLM auto → edit_region fallback
                if args.query:
                    text_query = args.query
                    print(f"  GroundingDINO query (CLI): \"{text_query}\"")
                else:
                    pil_for_step1 = Image.open(out_img_path).convert("RGB")
                    gsa_dir = REPO_ROOT / "data_generation" / "Grounded-Segment-Anything"
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    text_query = run_step1_query(pil_for_step1, meta, gsa_dir, device)
                    print(f"  GroundingDINO query: \"{text_query}\"")
                ok = run_grounded_sam(out_img_path, text_query, sample_dir,
                                      soft_mask_value=args.soft_mask_value)
                if not ok:
                    print(f"  [WARN] Grounded-SAM failed — masks not generated")
                elif "mask_post_process" in meta:
                    apply_mask_post_process(meta["mask_post_process"], sample_dir,
                                            soft_mask_value=args.soft_mask_value)
            else:
                print(f"  [WARN] No image to run SAM on")

        elif args.mask_mode == "ocr":
            if out_img_path.exists():
                # search_word: --query arg (cao nhất) → grounding_query → edit_region
                if args.query:
                    search_word = args.query
                else:
                    search_word = (meta.get("grounding_query")
                                   or meta.get("edit_region", "")).strip()
                # Nếu query dài (vd: "text LEFT") → lấy phần chữ in hoa cuối
                import re
                upper_words = re.findall(r'[A-Z]{2,}', search_word)
                if upper_words:
                    search_word = upper_words[-1]
                print(f"  OCR search word: \"{search_word}\"")
                ok = run_ocr_mask(out_img_path, search_word, sample_dir,
                                  soft_mask_value=args.soft_mask_value)
                if not ok:
                    print(f"  [WARN] OCR mask failed — masks not generated")
                elif "mask_post_process" in meta:
                    apply_mask_post_process(meta["mask_post_process"], sample_dir,
                                            soft_mask_value=args.soft_mask_value)
            else:
                print(f"  [WARN] No image to run OCR on")

        elif args.mask_mode == "manual":
            copy_manual_masks(args.manual_mf, args.manual_mb, sample_dir, IMAGE_SIZE)

        elif args.mask_mode == "skip":
            print(f"  Masks: skipped (add manually to {sample_dir}/Mf_mask.png + Mb_mask.png + Ms_mask.png)")

        # ── 3. Token count report ──────────────────────────────────────────
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            for key in ["source_caption_long", "source_caption_short",
                        "target_caption_long",  "target_caption_short", "instruction"]:
                n = len(enc.encode(meta[key]))
                if n > 248:
                    flag = "⚠️ >248 (exceeds LongCLIP)"
                elif n > 77:
                    flag = "⚠️ >77  (CLIP truncates here)"
                else:
                    flag = ""
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
