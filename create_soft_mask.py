import argparse
import os
import sys
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot load mask: {path}")
    return mask


def soft_mask_fusion(Mb, Mf, soft_value=0.5):
    """
    Mb: bounding box mask
    Mf: SAM fine mask
    """
    Mb_bin = (Mb > 128).astype(np.uint8)
    Mf_bin = (Mf > 128).astype(np.uint8)

    Ms = np.zeros_like(Mb_bin, dtype=np.float32)
    Ms[Mb_bin > 0] = soft_value
    Ms[Mf_bin > 0] = 1.0

    return (Ms * 255).astype(np.uint8)


# Alias used by run_eval_preprocess.py
def step5_soft_mask_fusion(Mb, Mf, soft_mask_value=0.5):
    return soft_mask_fusion(Mb, Mf, soft_value=soft_mask_value)


def smooth_edge_soft_mask(Mb, Mf):
    """Paper-style soft mask using distance transform"""
    Mb_bin = (Mb > 128).astype(np.uint8)
    Mf_bin = (Mf > 128).astype(np.uint8)

    dist_inside  = distance_transform_edt(Mf_bin)
    dist_outside = distance_transform_edt(Mb_bin - Mf_bin)
    dist = dist_inside + dist_outside
    dist = dist / dist.max()
    Ms = np.clip(dist, 0, 1)
    return (Ms * 255).astype(np.uint8)


# ── GroundingDINO helpers ──────────────────────────────────────────────────────
def load_groundingdino_model(config_path, checkpoint_path, device):
    """Load GroundingDINO model."""
    import torch
    _setup_gsa_path()
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)


def step3_grounding_dino(pil_image, target_object, gd_model, device,
                         box_threshold=0.3, text_threshold=0.25):
    """GroundingDINO → bounding box mask Mb."""
    import torch
    import numpy as np
    from torchvision.ops import box_convert
    _setup_gsa_path()
    import groundingdino.datasets.transforms as T
    from groundingdino.util.inference import predict

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_np = np.asarray(pil_image)
    image_transformed, _ = transform(pil_image, None)

    boxes, logits, phrases = predict(
        model=gd_model, image=image_transformed,
        caption=target_object,
        box_threshold=box_threshold, text_threshold=text_threshold,
    )

    if len(boxes) == 0:
        boxes, logits, phrases = predict(
            model=gd_model, image=image_transformed,
            caption=target_object,
            box_threshold=0.15, text_threshold=0.15,
        )

    if len(boxes) == 0:
        raise ValueError(f"GroundingDINO: cannot detect '{target_object}'")

    print(f"  Detected {len(boxes)} object(s): {phrases}")
    h, w = image_np.shape[:2]
    pixel_boxes = (boxes * torch.Tensor([w, h, w, h])).int()
    xyxy = box_convert(boxes=pixel_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    Mb = np.zeros((h, w), dtype=np.uint8)
    for box in xyxy:
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        Mb[y1:y2, x1:x2] = 255
        print(f"  Box: ({x1}, {y1}, {x2}, {y2})")

    return Mb, boxes, image_np


def step4_sam(image_np, boxes, sam_predictor, device):
    """SAM → fine-grained mask Mf."""
    import torch
    import numpy as np
    from torchvision.ops import box_convert

    sam_predictor.set_image(image_np)
    h, w = image_np.shape[:2]
    pixel_boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=pixel_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        xyxy.to(device), image_np.shape[:2]
    )
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None, point_labels=None,
        boxes=transformed_boxes, multimask_output=False,
    )
    merged = torch.max(masks.cpu(), 0)[0][0].numpy()
    Mf = (merged * 255).astype(np.uint8)
    print(f"  Mask coverage: {(Mf > 0).mean() * 100:.1f}%")
    return Mf


def _setup_gsa_path():
    """Add Grounded-Segment-Anything to sys.path if needed."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    gsa_dir   = os.path.join(repo_root, "data_generation", "Grounded-Segment-Anything")
    for p in [repo_root, gsa_dir, os.path.join(repo_root, "data_generation")]:
        if p not in sys.path:
            sys.path.insert(0, p)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox_mask", required=True)
    parser.add_argument("--sam_mask", required=True)
    parser.add_argument("--output_dir", default="mask_output")
    parser.add_argument("--mode", default="constant", choices=["constant", "distance"])
    parser.add_argument("--soft_value", type=float, default=0.5)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    Mb = load_mask(args.bbox_mask)
    Mf = load_mask(args.sam_mask)

    if args.mode == "constant":
        Ms = soft_mask_fusion(Mb, Mf, args.soft_value)
    else:
        Ms = smooth_edge_soft_mask(Mb, Mf)

    cv2.imwrite(os.path.join(args.output_dir, "Ms_soft_mask.png"), Ms)

    print("Saved:", os.path.join(args.output_dir, "Ms_soft_mask.png"))


if __name__ == "__main__":
    main()