"""
UltraEdit - Full 5-Step Soft Mask Pipeline
=============================================
Dùng GroundingDINO + SAM từ subrepo Grounded-Segment-Anything.

Cài đặt trước:
  # 1. Cài GroundingDINO
  cd data_generation/Grounded-Segment-Anything
  pip install -e GroundingDINO
  pip install segment-anything

  # 2. Tải model weights
  mkdir -p model/grounding_dino model/sam
  wget -P model/grounding_dino https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
  wget -P model/sam https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

  # 3. Chạy
  python create_soft_mask.py \
    --image images/example_images/Test_DOCCi.png \
    --target_object "ghost graffiti" \
    --output_dir mask_output
"""

import argparse
import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter, ImageDraw
from scipy.ndimage import distance_transform_edt

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data_generation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data_generation", "Grounded-Segment-Anything"))

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict
from segment_anything import build_sam, SamPredictor
from torchvision.ops import box_convert


def load_and_resize(image_path, target_size=512):
    """Load ảnh và resize về target_size x target_size (center crop)"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w > h:
        new_w = int(target_size * w / h)
        img = img.resize((new_w, target_size), Image.LANCZOS)
    else:
        new_h = int(target_size * h / w)
        img = img.resize((target_size, new_h), Image.LANCZOS)
    w, h = img.size
    left = (w - target_size) // 2
    top = (h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))
    return img


def gd_transform_image(pil_image):
    """Transform ảnh cho GroundingDINO input"""
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_np = np.asarray(pil_image)
    image_transformed, _ = transform(pil_image, None)
    return image_np, image_transformed


def load_groundingdino_model(config_path, checkpoint_path, device):
    """Load GroundingDINO model"""
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model = model.to(device)
    return model


# ============================================================
# BƯỚC 3: GroundingDINO → Bounding Box Mask (Mb)
# ============================================================
def step3_grounding_dino(pil_image, target_object, gd_model, device,
                         box_threshold=0.3, text_threshold=0.25):
    """Detect bounding box, trả về Mb và boxes"""
    print(f"\n[Bước 3] GroundingDINO: Detecting '{target_object}'...")
    
    image_np, image_transformed = gd_transform_image(pil_image)
    
    boxes, logits, phrases = predict(
        model=gd_model,
        image=image_transformed,
        caption=target_object,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if len(boxes) == 0:
        print("  WARNING: Không detect được, thử threshold thấp hơn...")
        boxes, logits, phrases = predict(
            model=gd_model,
            image=image_transformed,
            caption=target_object,
            box_threshold=0.15,
            text_threshold=0.15,
        )

    if len(boxes) == 0:
        raise ValueError(f"Không detect được '{target_object}' trong ảnh!")

    print(f"  Detected {len(boxes)} object(s): {phrases}")
    print(f"  Scores: {logits.tolist()}")

    # Tạo Mb từ tất cả detected boxes
    h, w = image_np.shape[:2]
    pixel_boxes = (boxes * torch.Tensor([w, h, w, h])).int()
    xyxy_boxes = box_convert(boxes=pixel_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    Mb = np.zeros((h, w), dtype=np.uint8)
    for box in xyxy_boxes:
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        Mb[y1:y2, x1:x2] = 255
        print(f"  Box: ({x1}, {y1}, {x2}, {y2})")

    return Mb, boxes, image_np


# ============================================================
# BƯỚC 4: SAM → Fine-grained Mask (Mf)
# ============================================================
def step4_sam(image_np, boxes, sam_predictor, device):
    """Dùng SAM segment object trong bounding box → Mf"""
    print(f"\n[Bước 4] SAM: Segmenting object...")

    sam_predictor.set_image(image_np)
    h, w = image_np.shape[:2]
    
    # Convert boxes sang xyxy format cho SAM
    pixel_boxes = (boxes * torch.Tensor([w, h, w, h]))
    xyxy_boxes = box_convert(boxes=pixel_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        xyxy_boxes.to(device), image_np.shape[:2]
    )

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Merge all masks
    merged_mask = torch.max(masks.cpu(), 0)[0][0].numpy()
    Mf = (merged_mask * 255).astype(np.uint8)

    print(f"  Mask coverage: {(Mf > 0).mean() * 100:.1f}%")
    return Mf


# ============================================================
# BƯỚC 5: Soft Mask Fusion → Ms
# ============================================================
def step5_soft_mask_fusion(Mb, Mf, soft_mask_value=0.5):
    """
    Fusion Mb + Mf → Soft Mask Ms 
    Như thực tế codebase UltraEdit: 
      - Vùng core Mf = 1.0 (trắng)
      - Vùng trong Mb nhưng ngoài Mf = soft_mask_value (xám)
      - Vùng ngoài Mb = 0.0 (đen)
    """
    print(f"\n[Bước 5] Soft Mask Fusion (Constant Alpha)...")

    Mf_binary = (Mf > 128).astype(np.float32)
    Mb_binary = (Mb > 128).astype(np.float32)

    Ms_final = np.zeros_like(Mf_binary, dtype=np.float32)
    
    # 1. Gán bounding box = soft_mask_value (VD: 0.5)
    Ms_final[Mb_binary > 0] = soft_mask_value
    
    # 2. Gán SAM mask = 1.0 (ghi đè lên vùng trong core)
    Ms_final[Mf_binary > 0] = 1.0

    print(f"  Core (=1.0): {(Ms_final == 1.0).mean() * 100:.1f}%")
    print(f"  Transition (={soft_mask_value}): {(Ms_final == soft_mask_value).mean() * 100:.1f}%")
    print(f"  Background (=0.0): {(Ms_final == 0.0).mean() * 100:.1f}%")

    return (Ms_final * 255).astype(np.uint8)


def create_preview(image, Mb, Mf, Ms, output_dir):
    """Tạo preview + comparison"""
    img_arr = np.array(image).astype(np.float32)
    Ms_norm = Ms.astype(np.float32) / 255.0
    overlay = img_arr.copy()
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + Ms_norm * 100, 0, 255)
    overlay[:, :, 0] = overlay[:, :, 0] * (1 - Ms_norm * 0.4)
    Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(output_dir, "preview_overlay.png"))

    w, h = image.size
    comp = Image.new("RGB", (w * 4, h))
    comp.paste(image, (0, 0))
    comp.paste(Image.fromarray(np.stack([Mb]*3, axis=-1)), (w, 0))
    comp.paste(Image.fromarray(np.stack([Mf]*3, axis=-1)), (w*2, 0))
    comp.paste(Image.fromarray(np.stack([Ms]*3, axis=-1)), (w*3, 0))
    draw = ImageDraw.Draw(comp)
    for i, label in enumerate(["Original", "Mb (BBox)", "Mf (SAM)", "Ms (Soft)"]):
        draw.rectangle([(i*w, 0), (i*w + 120, 22)], fill="black")
        draw.text((i*w + 5, 3), label, fill="white")
    comp.save(os.path.join(output_dir, "comparison_all.png"))


def main():
    parser = argparse.ArgumentParser(description="UltraEdit 5-Step Soft Mask Pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--target_object", required=True, help="Object cần mask, vd: 'ghost graffiti'")
    parser.add_argument("--output_dir", default="mask_output")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--soft_mask_value", type=float, default=0.5, help="Giá trị xám cho Box Mask (0.0 -> 1.0)")
    parser.add_argument("--groundingdino_config",
                        default="data_generation/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py")
    parser.add_argument("--groundingdino_checkpoint",
                        default="data_generation/Grounded-Segment-Anything/model/grounding_dino/groundingdino_swinb_cogcoor.pth")
    parser.add_argument("--sam_checkpoint",
                        default="data_generation/Grounded-Segment-Anything/model/sam/sam_vit_h_4b8939.pth")
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load image
    print(f"Loading: {args.image}")
    image = load_and_resize(args.image, args.image_size)
    image.save(os.path.join(args.output_dir, "input_resized.png"))

    # Bước 1-2: skip (target đã biết)
    print(f"\n[Bước 1-2] Target: '{args.target_object}'")

    # Bước 3: GroundingDINO
    print(f"Loading GroundingDINO from: {args.groundingdino_checkpoint}")
    gd_model = load_groundingdino_model(args.groundingdino_config, args.groundingdino_checkpoint, device)
    Mb, boxes, image_np = step3_grounding_dino(image, args.target_object, gd_model, device,
                                                args.box_threshold, args.text_threshold)
    Image.fromarray(Mb).save(os.path.join(args.output_dir, "Mb_bounding_box.png"))
    del gd_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # Bước 4: SAM
    print(f"Loading SAM from: {args.sam_checkpoint}")
    sam = build_sam(checkpoint=args.sam_checkpoint).to(device)
    sam_predictor = SamPredictor(sam)
    Mf = step4_sam(image_np, boxes, sam_predictor, device)
    Image.fromarray(Mf).save(os.path.join(args.output_dir, "Mf_fine_grained.png"))
    del sam, sam_predictor
    torch.cuda.empty_cache() if device == "cuda" else None

    # Bước 5: Soft Mask
    Ms = step5_soft_mask_fusion(Mb, Mf, args.soft_mask_value)
    Image.fromarray(Ms).save(os.path.join(args.output_dir, "Ms_soft_mask.png"))
    Image.merge("RGB", [Image.fromarray(Ms)] * 3).save(os.path.join(args.output_dir, "Ms_soft_mask_rgb.png"))

    # Preview
    create_preview(image, Mb, Mf, Ms, args.output_dir)

    print(f"\n{'='*50}")
    print(f"Results saved to: {args.output_dir}/")
    print(f"  Mb_bounding_box.png  → GroundingDINO")
    print(f"  Mf_fine_grained.png  → SAM")
    print(f"  Ms_soft_mask.png     → Soft Mask")
    print(f"  Ms_soft_mask_rgb.png → RGB (cho pipeline)")
    print(f"  comparison_all.png   → So sánh 4 ảnh")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
