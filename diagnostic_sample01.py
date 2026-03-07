import sys, torch
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, 'data_generation/Grounded-Segment-Anything')
from groundingdino.util.inference import load_model, load_image, predict

cfg  = 'data_generation/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py'
ckpt = 'data_generation/Grounded-Segment-Anything/model/grounding_dino/groundingdino_swinb_cogcoor.pth'
model = load_model(cfg, ckpt)

img_path = 'eval_data/sample_01/image.png'
image_source, image_tensor = load_image(img_path)
H, W, _ = image_source.shape
print(f'Image size: {W}x{H}')

queries = [
    'graffiti text CHINA on concrete overpass',
    'CHINA',
    'graffiti letters',
    'text on overpass',
    'overpass graffiti',
    'concrete overpass bar',
]

best_query = None
best_boxes = None

for query in queries:
    boxes, logits, phrases = predict(
        model=model, image=image_tensor,
        caption=query, box_threshold=0.25, text_threshold=0.20)
    if len(boxes) > 0:
        boxes_px = boxes * torch.tensor([W, H, W, H], dtype=torch.float32)
        cx, cy, w, h = boxes_px[:,0], boxes_px[:,1], boxes_px[:,2], boxes_px[:,3]
        x1, y1, x2, y2 = (cx-w/2).int(), (cy-h/2).int(), (cx+w/2).int(), (cy+h/2).int()
        print(f'\nQuery "{query}": {len(boxes)} boxes')
        for i in range(len(boxes)):
            print(f'  [{i}] phrase={phrases[i]!r} conf={logits[i]:.2f}  bbox=({x1[i].item()},{y1[i].item()})-({x2[i].item()},{y2[i].item()})')
        if best_boxes is None:
            best_query = query
            best_boxes = (boxes, x1, y1, x2, y2, phrases)
    else:
        print(f'\nQuery "{query}": 0 boxes')

# Visualize best result
if best_boxes is not None:
    boxes, x1, y1, x2, y2, phrases = best_boxes
    img_vis = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img_vis)
    for i in range(len(boxes)):
        draw.rectangle([x1[i].item(), y1[i].item(), x2[i].item(), y2[i].item()],
                       outline='red', width=3)
        draw.text((x1[i].item(), max(0, y1[i].item()-15)), phrases[i], fill='red')
    img_vis.save('eval_data/sample_01/debug_boxes.png')
    print(f'\nVisualization saved to eval_data/sample_01/debug_boxes.png')
    print(f'Best query: "{best_query}"')
