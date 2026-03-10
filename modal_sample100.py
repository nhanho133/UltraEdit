"""
Modal — 100-seed GT Replication
================================
Chạy đúng như UltraEdit data_generation pipeline:
  - Model    : stabilityai/sdxl-turbo (4-ch, SAME as GT)
  - Pipeline : Prompt2PromptInpaintPipeline (SAME as GT)
  - 100 random seeds, mỗi seed random hóa:
      * p2p_threshold  ∈ uniform(0.1, 0.9)
      * steps          ∈ {10, 14}
      * guidance_scale ∈ {0.0, 0.2, 0.4, 0.6}
      * soft_mask      ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.8}
      * using_soft_mask∈ {True, False}
  - Filter bằng đúng thresholds từ run_inpainting_multiiple_objects.sh:
      * clip_sim_image >= 0.70  (source vs edited, image-image cosine sim)
      * clip_sim_dir   >= 0.22  (directional CLIP)
      * clip_sim_0     >= 0.20  (CLIP text-image, source caption vs source img)
      * clip_sim_1     >= 0.20  (CLIP text-image, target caption vs edited img)
      * dinov2_sim     >= 0.40  (DINOv2 cosine sim, source vs edited)
  - Sort descending by clip_sim_dir → lấy top-3 (max_out_samples=3)
  - Trả về best image + toàn bộ metrics scatter plot

Usage:
  modal run modal_sample100.py::main \\
    --image-path eval_data/sample_ue/image.png \\
    --mf-mask    eval_data/sample_ue/Mf_mask.png \\
    --mb-mask    eval_data/sample_ue/Mb_mask.png \\
    --source-caption "Two giraffes are next to a tall tree." \\
    --target-caption "Two giraffes are next to a colourful rainbow tree." \\
    --out-dir    region_output_100seeds
"""

import modal
import os
import io
import sys
from pathlib import Path

app = modal.App("ultraedit-100seeds")

# ── Docker Image — same stack as modal_region_edit.py ─────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget", "libgl1-mesa-glx", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev",
    )
    .pip_install("numpy<2")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.36.2",
        "accelerate>=0.24.0",
        "huggingface_hub>=0.19.0",
        "safetensors",
    )
    .pip_install(
        "Pillow>=9.0.0",
        "opencv-python-headless",
        "open-clip-torch>=2.20.0",
        "scipy",
        "ftfy",
        "regex",
        "tqdm",
        "einops",
        "scikit-image",     # for SSIM
        "openai-clip",      # for CLIP ViT-L/14
        "matplotlib",       # for scatter plot
    )
    .run_commands(
        "git clone https://github.com/HaozheZhao/UltraEdit.git /repo/UltraEdit",
        "git clone https://github.com/beichenzbc/Long-CLIP.git /repo/UltraEdit/Long-CLIP",
        "cd /repo/UltraEdit/Long-CLIP && pip install -r requirements.txt 2>/dev/null || true",
        "cd /repo/UltraEdit/diffusers && pip install -e . --no-deps 2>/dev/null || true",
    )
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_generation"),
        remote_path="/repo/UltraEdit/data_generation",
    )
)

model_cache = modal.Volume.from_name("ultraedit-model-cache", create_if_missing=True)
CACHE_DIR  = "/cache"
HF_CACHE   = "/cache/huggingface"
GPU_CONFIG = "A10G"

# ── Filter thresholds (từ run_inpainting_multiiple_objects.sh) ────────────────
CLIP_IMG_THRESH = 0.70   # clip_sim_image (source vs edited image-image)
CLIP_DIR_THRESH = 0.22   # directional CLIP
CLIP_THRESH     = 0.20   # clip_sim_0 and clip_sim_1 (text-image)
DINOV2_THRESH   = 0.40   # DINOv2 cosine sim
MAX_OUT_SAMPLES = 3      # top-k survivors


# ═════════════════════════════════════════════════════════════════════════════
# REMOTE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: model_cache},
    timeout=1800,
    memory=20480,
)
def run_100_seeds(
    image_bytes:    bytes,
    mf_mask_bytes:  bytes,
    mb_mask_bytes:  bytes,
    source_caption: str,
    target_caption: str,
    n_seeds:        int   = 100,
    pipeline_ckpt:  str   = "stabilityai/sdxl-turbo",
    image_size:     int   = 512,
    # p2p range (GT: min_p2p=0.1, max_p2p=0.9 from shell script)
    min_p2p:        float = 0.1,
    max_p2p:        float = 0.9,
) -> dict:
    import torch
    import random
    import numpy as np
    import warnings
    import logging
    import clip
    import torch.nn.functional as F
    from PIL import Image
    from skimage.metrics import structural_similarity as sk_ssim

    warnings.filterwarnings("ignore")
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    _orig = logging.Logger.warning
    def _filtered(self, msg, *a, **kw):
        if "cross_attention_kwargs" in str(msg) and "not expected" in str(msg):
            return
        _orig(self, msg, *a, **kw)
    logging.Logger.warning = _filtered

    device = "cuda"
    dtype  = torch.float16

    print(f"\n{'='*60}")
    print(f"  UltraEdit 100-seed GT Replication")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  n_seeds={n_seeds} | model={pipeline_ckpt}")
    print(f"{'='*60}")

    sys.path.insert(0, "/repo/UltraEdit/diffusers/src")
    sys.path.insert(0, "/repo/UltraEdit/data_generation")
    from sdxl_p2p_pipeline import Prompt2PromptInpaintPipeline

    # ── Load image + masks ─────────────────────────────────────────────────────
    def bytes_to_pil(b, mode="RGB"):
        return Image.open(io.BytesIO(b)).convert(mode)
    def resize_center(img, size):
        w, h = img.size
        scale = size / min(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        w, h = img.size
        return img.crop(((w-size)//2, (h-size)//2, (w+size)//2, (h+size)//2))

    src_img = resize_center(bytes_to_pil(image_bytes), image_size)
    mf_mask = bytes_to_pil(mf_mask_bytes, "L").resize((image_size, image_size), Image.NEAREST)
    mb_mask = bytes_to_pil(mb_mask_bytes, "L").resize((image_size, image_size), Image.NEAREST)

    src_t = torch.from_numpy(np.array(src_img)).permute(2,0,1).float() / 255.0  # [3,H,W]

    # ── Load Diffusion Pipeline ────────────────────────────────────────────────
    print("\n[1] Loading SDXL-Turbo P2P pipeline...")
    pipe = Prompt2PromptInpaintPipeline.from_pretrained(
        pipeline_ckpt,
        torch_dtype=dtype,
        variant="fp16",
        cache_dir=HF_CACHE,
    ).to(device)
    pipe.unet.config.addition_embed_type = None   # exactly like GT
    pipe.set_progress_bar_config(disable=True)
    print(f"  UNet in_channels={pipe.unet.config.in_channels} ✓")

    # ── Load CLIP ViT-L/14 ─────────────────────────────────────────────────────
    print("\n[2] Loading CLIP ViT-L/14...")
    clip_model, _ = clip.load("ViT-L/14", device=device, download_root=HF_CACHE)
    clip_model.eval().requires_grad_(False)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(3,1,1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(3,1,1)

    def clip_encode_img(img_t):
        """img_t: [3,H,W] float [0,1] on CPU → CLIP feature [1,768]"""
        x = img_t.unsqueeze(0).to(device)
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
        x = (x - clip_mean) / clip_std
        with torch.no_grad():
            feat = clip_model.encode_image(x.half())
        return (feat / feat.norm(dim=-1, keepdim=True)).float()

    def clip_encode_txt(text):
        toks = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            feat = clip_model.encode_text(toks)
        return (feat / feat.norm(dim=-1, keepdim=True)).float()

    # ── Load DINOv2 ViT-L/14 reg ──────────────────────────────────────────────
    print("\n[3] Loading DINOv2 ViT-L/14-reg...")
    dinov2 = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14_reg",
        source="github",
    ).to(device).eval().requires_grad_(False)

    def dinov2_encode(img_t):
        """img_t: [3,H,W] float [0,1] on CPU → DINOv2 feature [1,1024]"""
        x = img_t.unsqueeze(0).to(device)
        x = F.interpolate(x, size=(518, 518), mode="bicubic", align_corners=False)
        with torch.no_grad():
            feat = dinov2(x.float())
        return (feat / feat.norm(dim=-1, keepdim=True)).float()

    # Pre-compute source features (fixed across all seeds)
    print("\n[4] Pre-computing source features...")
    src_clip_feat  = clip_encode_img(src_t)
    src_dino_feat  = dinov2_encode(src_t)
    txt0_feat      = clip_encode_txt(source_caption)
    txt1_feat      = clip_encode_txt(target_caption)
    print(f"  Done. CLIP img: {src_clip_feat.shape}  DINOv2: {src_dino_feat.shape}")

    # ── P2P captions ──────────────────────────────────────────────────────────
    src_cap = source_caption
    tgt_cap = target_caption
    diff = [b for a, b in zip(src_cap.split(), tgt_cap.split()) if a != b]
    edit_type = "replace" if (diff and len(src_cap.split()) == len(tgt_cap.split())) else "refine"

    # ── 100-seed loop ──────────────────────────────────────────────────────────
    print(f"\n[5] Running {n_seeds} random seeds...")
    results = []

    for i in range(n_seeds):
        # Random seed — exactly like GT main loop
        seed = torch.randint(1 << 32, ()).item()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Random hyperparams — exactly like generate_images() for inpainting
        p2p_thr = min_p2p + random.random() * (max_p2p - min_p2p)  # uniform [0.1, 0.9]
        steps   = random.choice([10, 14])
        cfg     = random.choice([0.0, 0.2, 0.4, 0.6])
        soft    = random.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.8])
        use_soft = random.choice([True, False])

        # P2P cross-attention kwargs
        n_cross = ({"default_": 1.0, " ".join(diff[:5]): p2p_thr}
                   if edit_type == "replace" and diff else p2p_thr)
        cross_attn = {
            "edit_type":       edit_type,
            "n_self_replace":  p2p_thr,
            "n_cross_replace": n_cross,
            "prompts":         [src_cap, tgt_cap],
            "max_num_words":   77,
        }

        call_kw = dict(
            prompt              = [src_cap, tgt_cap],
            image               = src_img,
            mask_image          = mf_mask,
            temp_mask           = mb_mask,
            num_inference_steps = steps,
            guidance_scale      = cfg,
            cross_attention_kwargs = cross_attn,
            output_type         = "pil",
        )
        if use_soft:
            call_kw["soft_mask"]  = soft
            call_kw["mask_choice"] = "wo_final_layer"

        try:
            out = pipe(**call_kw).images
            edited_img = out[1]   # index 0 = src recon, 1 = target
        except Exception as e:
            print(f"  seed {i}: FAILED — {e}")
            continue

        # ── Compute metrics ────────────────────────────────────────────────────
        edit_t = torch.from_numpy(np.array(edited_img)).permute(2,0,1).float() / 255.0

        edit_clip_feat = clip_encode_img(edit_t)
        edit_dino_feat = dinov2_encode(edit_t)

        clip_sim_0     = F.cosine_similarity(src_clip_feat,  txt0_feat).item()   # src img vs src text
        clip_sim_1     = F.cosine_similarity(edit_clip_feat, txt1_feat).item()   # edit img vs tgt text
        clip_sim_image = F.cosine_similarity(src_clip_feat,  edit_clip_feat).item()  # img-img
        clip_sim_dir   = F.cosine_similarity(                                    # directional
            edit_clip_feat - src_clip_feat,
            txt1_feat      - txt0_feat
        ).item()
        dinov2_sim     = F.cosine_similarity(src_dino_feat, edit_dino_feat).item()

        # SSIM (on CPU)
        src_np  = np.array(src_img).astype(np.float32) / 255.0
        edit_np = np.array(edited_img).astype(np.float32) / 255.0
        ssim_val = np.mean([sk_ssim(src_np[:,:,c], edit_np[:,:,c], data_range=1.0) for c in range(3)])

        rec = dict(
            seed          = seed,
            idx           = i,
            p2p_thr       = p2p_thr,
            steps         = steps,
            cfg           = cfg,
            soft          = soft if use_soft else None,
            use_soft      = use_soft,
            clip_sim_0    = clip_sim_0,
            clip_sim_1    = clip_sim_1,
            clip_sim_image= clip_sim_image,
            clip_sim_dir  = clip_sim_dir,
            dinov2_sim    = dinov2_sim,
            ssim          = ssim_val,
            pass_filter   = (
                clip_sim_image >= CLIP_IMG_THRESH and
                clip_sim_dir   >= CLIP_DIR_THRESH and
                clip_sim_0     >= CLIP_THRESH     and
                clip_sim_1     >= CLIP_THRESH     and
                dinov2_sim     >= DINOV2_THRESH
            ),
        )
        if rec["pass_filter"]:
            # Store image bytes only for survivors to save memory
            buf = io.BytesIO(); edited_img.save(buf, format="PNG")
            rec["image_bytes"] = buf.getvalue()
            buf0 = io.BytesIO(); out[0].save(buf0, format="PNG")
            rec["src_recon_bytes"] = buf0.getvalue()

        results.append(rec)

        if (i+1) % 10 == 0:
            n_pass = sum(r["pass_filter"] for r in results)
            print(f"  [{i+1:3d}/{n_seeds}]  pass={n_pass}  "
                  f"best_dir={max((r['clip_sim_dir'] for r in results), default=0):.3f}")

    # ── Filter & rank ──────────────────────────────────────────────────────────
    survivors = [r for r in results if r["pass_filter"]]
    survivors.sort(key=lambda r: r["clip_sim_dir"], reverse=True)
    top = survivors[:MAX_OUT_SAMPLES]

    print(f"\n[6] Results: {len(survivors)}/{n_seeds} passed filter → top {len(top)} saved")
    for j, r in enumerate(top):
        print(f"  #{j+1}  seed={r['seed']}  dir={r['clip_sim_dir']:.3f}  "
              f"img={r['clip_sim_image']:.3f}  dino={r['dinov2_sim']:.3f}  "
              f"steps={r['steps']}  cfg={r['cfg']}  soft={r['soft']}")

    return {
        "all_results":  results,     # all 100 records (no image_bytes for non-survivors)
        "top":          top,         # top-3 with image_bytes
        "n_pass":       len(survivors),
        "n_seeds":      n_seeds,
    }


# ═════════════════════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    image_path:     str   = "eval_data/sample_ue/image.png",
    mf_mask:        str   = "eval_data/sample_ue/Mf_mask.png",
    mb_mask:        str   = "eval_data/sample_ue/Mb_mask.png",
    source_caption: str   = "Two giraffes are next to a tall tree.",
    target_caption: str   = "Two giraffes are next to a colourful rainbow tree.",
    n_seeds:        int   = 100,
    out_dir:        str   = "region_output_100seeds",
    pipeline_ckpt:  str   = "stabilityai/sdxl-turbo",
):
    import json
    import numpy as np
    from PIL import Image, ImageDraw

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  UltraEdit 100-seed GT Replication")
    print(f"  Image  : {image_path}")
    print(f"  Source : {source_caption}")
    print(f"  Target : {target_caption}")
    print(f"  Seeds  : {n_seeds}")
    print(f"  Thresholds: clip_img>={CLIP_IMG_THRESH} dir>={CLIP_DIR_THRESH} "
          f"dino>={DINOV2_THRESH}")
    print(f"{'='*60}")

    result = run_100_seeds.remote(
        image_bytes     = Path(image_path).read_bytes(),
        mf_mask_bytes   = Path(mf_mask).read_bytes(),
        mb_mask_bytes   = Path(mb_mask).read_bytes(),
        source_caption  = source_caption,
        target_caption  = target_caption,
        n_seeds         = n_seeds,
        pipeline_ckpt   = pipeline_ckpt,
    )

    all_res = result["all_results"]
    top     = result["top"]
    n_pass  = result["n_pass"]

    # ── Save top images ────────────────────────────────────────────────────────
    SIZE = 512
    src   = Image.open(image_path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    mf_v  = Image.open(mf_mask).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    gt_path = Path(image_path).parent / "gt_edited.png"
    gt_img  = Image.open(gt_path).convert("RGB").resize((SIZE, SIZE)) if gt_path.exists() else None

    print(f"\n[Results] {n_pass}/{n_seeds} passed filter")
    for j, r in enumerate(top):
        tgt = Image.open(io.BytesIO(r["image_bytes"])).convert("RGB")
        tag = f"top{j+1}_seed{r['seed']}_dir{r['clip_sim_dir']:.3f}_steps{r['steps']}_cfg{r['cfg']}"
        tgt.save(out_dir / f"{tag}.png")

        # Comparison panel
        panels = [(src, "Source"), (mf_v.convert("RGB"), "Mask")]
        if gt_img:
            panels.append((gt_img, "Dataset GT"))
        panels.append((tgt, f"#{j+1} dir={r['clip_sim_dir']:.3f}\nsteps={r['steps']} cfg={r['cfg']} soft={r['soft']}"))

        W = SIZE * len(panels)
        comp = Image.new("RGB", (W, SIZE), (20, 20, 20))
        draw = ImageDraw.Draw(comp)
        for i, (im, lbl) in enumerate(panels):
            comp.paste(im.resize((SIZE, SIZE)), (i * SIZE, 0))
            first_line = lbl.split("\n")[0]
            draw.rectangle([(i*SIZE+2, 2), (i*SIZE + len(first_line)*8+8, 22)], fill=(0,0,0))
            draw.text((i*SIZE+5, 4), first_line, fill="white")
            if "\n" in lbl:
                draw.text((i*SIZE+5, 20), lbl.split("\n")[1], fill=(200,200,200))
        comp.save(out_dir / f"comparison_top{j+1}.png")
        print(f"  #{j+1}: clip_dir={r['clip_sim_dir']:.3f}  clip_img={r['clip_sim_image']:.3f}  "
              f"dino={r['dinov2_sim']:.3f}  steps={r['steps']}  cfg={r['cfg']}  soft={r['soft']}")

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    # Remove image bytes before saving JSON
    clean = [{k:v for k,v in r.items() if k not in ("image_bytes","src_recon_bytes")}
             for r in all_res]
    with open(out_dir / "all_metrics.json", "w") as f:
        json.dump(clean, f, indent=2)

    # ── Print summary stats ────────────────────────────────────────────────────
    dirs   = [r["clip_sim_dir"]   for r in all_res]
    imgs   = [r["clip_sim_image"] for r in all_res]
    dinos  = [r["dinov2_sim"]     for r in all_res]

    print(f"\n[Metric distribution over {n_seeds} seeds]")
    print(f"  clip_sim_dir  : mean={np.mean(dirs):.3f}  max={np.max(dirs):.3f}  "
          f"≥{CLIP_DIR_THRESH}: {sum(d>=CLIP_DIR_THRESH for d in dirs)}")
    print(f"  clip_sim_image: mean={np.mean(imgs):.3f}  max={np.max(imgs):.3f}  "
          f"≥{CLIP_IMG_THRESH}: {sum(d>=CLIP_IMG_THRESH for d in imgs)}")
    print(f"  dinov2_sim    : mean={np.mean(dinos):.3f}  max={np.max(dinos):.3f}  "
          f"≥{DINOV2_THRESH}: {sum(d>=DINOV2_THRESH for d in dinos)}")
    print(f"  TOTAL PASS    : {n_pass}/{n_seeds}")

    # ── Scatter plot clip_sim_dir vs dinov2_sim ───────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pass_r = [r for r in all_res if r["pass_filter"]]
        fail_r = [r for r in all_res if not r["pass_filter"]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        if fail_r:
            ax.scatter([r["clip_sim_dir"] for r in fail_r],
                       [r["dinov2_sim"] for r in fail_r],
                       c="lightcoral", alpha=0.5, s=20, label="fail")
        if pass_r:
            ax.scatter([r["clip_sim_dir"] for r in pass_r],
                       [r["dinov2_sim"] for r in pass_r],
                       c="steelblue", alpha=0.8, s=40, label="pass")
        for j, r in enumerate(top):
            ax.scatter(r["clip_sim_dir"], r["dinov2_sim"],
                       c="gold", s=150, marker="*", zorder=5,
                       label=f"top-{j+1}" if j==0 else "")
        ax.axvline(CLIP_DIR_THRESH, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(DINOV2_THRESH,   color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("clip_sim_dir (directional CLIP)")
        ax.set_ylabel("dinov2_sim")
        ax.set_title(f"100-seed scatter: {n_pass}/{n_seeds} pass")
        ax.legend()

        ax2 = axes[1]
        if fail_r:
            ax2.scatter([r["clip_sim_image"] for r in fail_r],
                        [r["clip_sim_dir"] for r in fail_r],
                        c="lightcoral", alpha=0.5, s=20, label="fail")
        if pass_r:
            ax2.scatter([r["clip_sim_image"] for r in pass_r],
                        [r["clip_sim_dir"] for r in pass_r],
                        c="steelblue", alpha=0.8, s=40, label="pass")
        for j, r in enumerate(top):
            ax2.scatter(r["clip_sim_image"], r["clip_sim_dir"],
                        c="gold", s=150, marker="*", zorder=5)
        ax2.axvline(CLIP_IMG_THRESH, color="gray", linestyle="--", linewidth=0.8)
        ax2.axhline(CLIP_DIR_THRESH, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("clip_sim_image (image-image)")
        ax2.set_ylabel("clip_sim_dir")
        ax2.set_title("CLIP image-image vs directional")
        ax2.legend()

        plt.suptitle(f"UltraEdit 100-seed filtering | sample_ue\n"
                     f"src: '{source_caption[:50]}'\ntgt: '{target_caption[:50]}'",
                     fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "scatter.png", dpi=150)
        print(f"\n[Saved] {out_dir}/scatter.png")
    except Exception as e:
        print(f"  [scatter plot failed: {e}]")

    print(f"\n[Saved]  {out_dir}/")
    print(f"  top*.png  comparison_top*.png  scatter.png  all_metrics.json")
    print(f"\n{'='*60}  Done!\n")
