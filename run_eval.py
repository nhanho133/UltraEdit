"""
run_eval.py
===========
Batch evaluation runner — 5 experiments × 5 DOCCI samples.

Experiments:
  1_clip77         — SDXL-Turbo + CLIP 77t,     long captions (truncated at 77)
  2_longclip248    — SDXL-Turbo + Long-CLIP 248t, long captions
  3_longclip_short — SDXL-Turbo + Long-CLIP 248t, short captions (ablation: encoder vs length)
  4_sd3_short      — SD3 UltraEdit,              short instruction only
  5_sd3_long       — SD3 UltraEdit,              src_long + instruction (T5-XXL 256t)

Usage:
  # All 5 experiments, all 5 samples:
  modal run run_eval.py

  # Specific experiments only:
  modal run run_eval.py --experiments 1_clip77 4_sd3_short

  # Specific samples only:
  modal run run_eval.py --samples sample_01 sample_03

  # Single experiment, single sample (debug):
  modal run run_eval.py --experiments 1_clip77 --samples sample_01 --seed 0
"""

import modal
import os
import io
import json
import time
import argparse
from pathlib import Path
from typing import Optional

# ── Modal function handles (resolved lazily inside dispatch_job) ──────────────
# These are looked up per-call so a missing deployment only errors when that
# experiment is actually dispatched, not at import time.
# Deploy before running:
#   modal deploy modal_region_edit.py
#   modal deploy modal_sd3_edit.py

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent
EVAL_DATA    = REPO_ROOT / "eval_data"
EVAL_RESULTS = REPO_ROOT / "eval_results"

# ── Experiment definitions ─────────────────────────────────────────────────────
#
# name            : output folder name
# modal_fn        : which Modal remote function to call
# use_long_clip   : SDXL experiments only
# caption_mode    : "long" | "short"  — which src/tgt caption variant to use
# sd3_prompt_mode : "instruction" | "long_src+instruction" — SD3 prompt construction

EXPERIMENTS = [
    {
        "name":           "1_clip77",
        "description":    "SDXL-Turbo + CLIP 77t, long captions (77-token truncation baseline)",
        "modal_fn":       "sdxl",
        "use_long_clip":  False,
        "caption_mode":   "long",
    },
    {
        "name":             "2_longclip248",
        "description":      "SDXL-Turbo + LongCLIP-GmP 248t, long captions (CLIP-G zeroed)",
        "modal_fn":         "sdxl",
        "use_long_clip":    True,
        "longclip_encoder": "zer0int",
        "caption_mode":     "long",
    },
    {
        "name":             "3_longclip_short",
        "description":      "SDXL-Turbo + LongCLIP-GmP 248t, short captions (ablation: same encoder, short text)",
        "modal_fn":         "sdxl",
        "use_long_clip":    True,
        "longclip_encoder": "zer0int",
        "caption_mode":     "short",
    },
    {
        "name":           "4_clip77_short",
        "description":    "SDXL-Turbo + CLIP 77t, short captions (ablation: same encoder as exp1, short text)",
        "modal_fn":       "sdxl",
        "use_long_clip":  False,
        "caption_mode":   "short",
    },
    {
        "name":           "5_sd3_short",
        "description":    "SD3 UltraEdit, short instruction only",
        "modal_fn":       "sd3",
        "sd3_prompt_mode": "instruction",
    },
    {
        "name":           "6_sd3_long",
        "description":    "SD3 UltraEdit, src_long + instruction (T5-XXL, up to 256t)",
        "modal_fn":       "sd3",
        "sd3_prompt_mode": "long_src+instruction",
    },
]

# ── SDXL hyperparams ───────────────────────────────────────────────────────────
# mask_choice controls when background-preservation is applied during denoising:
#   None              → only odd timesteps (default, 1/3 enforcements with steps=6,strength=0.5)
#   "wo_final_layer"  → all steps except last (2/3 enforcements) → better BG preservation
#   "wo_final_two_layer" → all except last 2 steps
# With steps=6, strength=0.5 → 3 actual denoising steps (idx 0,1,2).
# Default (odd-only) gives 1 enforcement → BG PSNR degrades to 17-21dB for large-mask samples.
# "wo_final_layer" gives 2 enforcements → significantly better background fidelity.
SDXL_PARAMS = dict(
    soft_mask_value = 0.5,
    p2p_threshold   = 0.7,
    steps           = 6,
    strength        = 0.5,
    guidance_scale  = 0.0,   # SDXL-Turbo = CFG-free
    mask_choice     = "wo_final_layer",  # enforce BG at steps 0,1 (not last) → better BG fidelity
)

# ── SD3 hyperparams ────────────────────────────────────────────────────────────
SD3_PARAMS = dict(
    steps           = 28,
    guidance_scale  = 7.0,
    image_guidance  = 1.5,
)


# ── Helper: load eval sample ───────────────────────────────────────────────────
def load_sample(sample_dir: Path) -> dict:
    """Load metadata + images/masks → bytes."""
    with open(sample_dir / "metadata.json") as f:
        meta = json.load(f)

    def read(path, required=True):
        p = sample_dir / path
        if not p.exists():
            if required:
                raise FileNotFoundError(f"Required file missing: {p}")
            return None
        with open(p, "rb") as f:
            return f.read()

    return {
        "meta":         meta,
        "image_bytes":  read("image.png"),
        "mf_bytes":     read("Mf_mask.png",  required=False),
        "mb_bytes":     read("Mb_mask.png",  required=False),
    }


# ── Helper: dummy mask fallback ────────────────────────────────────────────────
def make_white_mask_bytes(size=512) -> bytes:
    """Fallback: all-white mask (edit entire image) nếu mask chưa được sinh."""
    from PIL import Image as PILImage
    import numpy as np
    buf = io.BytesIO()
    PILImage.fromarray(np.ones((size, size), dtype=np.uint8) * 255).save(buf, format="PNG")
    return buf.getvalue()


# ── Helper: save output + metadata ────────────────────────────────────────────
def save_result(exp_name: str, sample_id: str, result: dict,
                meta_used: dict, elapsed: float):
    out_dir = EVAL_RESULTS / exp_name / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image as PILImage

    if "target" in result:
        PILImage.open(io.BytesIO(result["target"])).save(out_dir / "target.png")
    if "source_recon" in result:
        PILImage.open(io.BytesIO(result["source_recon"])).save(out_dir / "source_recon.png")

    meta_used["elapsed_sec"] = round(elapsed, 1)
    with open(out_dir / "metadata_used.json", "w") as f:
        json.dump(meta_used, f, indent=2, ensure_ascii=False)

    print(f"  → Saved to eval_results/{exp_name}/{sample_id}/ ({elapsed:.1f}s)")


# ── Dispatch single job (non-blocking, returns future + metadata) ──────────────
def dispatch_job(exp: dict, sample: dict, seed: int = 42):
    """Spawn a Modal job and return (future, meta_used, t0) without blocking."""
    meta     = sample["meta"]
    mf_bytes = sample["mf_bytes"] or make_white_mask_bytes()
    mb_bytes = sample["mb_bytes"] or make_white_mask_bytes()
    t0       = time.time()

    # Lazy lookup — only resolves the app that's actually needed for this job
    run_region_edit = modal.Function.from_name("ultraedit-region-edit", "run_region_edit")

    if exp["modal_fn"] == "sdxl":
        if exp["caption_mode"] == "long":
            src_cap = meta["source_caption_long"]
            tgt_cap = meta["target_caption_long"]
        else:
            src_cap = meta["source_caption_short"]
            tgt_cap = meta["target_caption_short"]

        fut = run_region_edit.spawn(
            image_bytes      = sample["image_bytes"],
            mf_mask_bytes    = mf_bytes,
            mb_mask_bytes    = mb_bytes,
            source_caption   = src_cap,
            target_caption   = tgt_cap,
            use_long_clip    = exp["use_long_clip"],
            longclip_encoder = exp.get("longclip_encoder", "beichen"),
            seed             = seed,
            **SDXL_PARAMS,
        )
        meta_used = {
            "experiment":        exp["name"],
            "description":       exp["description"],
            "use_long_clip":     exp["use_long_clip"],
            "longclip_encoder":  exp.get("longclip_encoder", "beichen"),
            "caption_mode":      exp["caption_mode"],
            "source_caption":    src_cap,
            "target_caption":    tgt_cap,
            "source_token_est":  len(src_cap.split()),
            "target_token_est":  len(tgt_cap.split()),
            "sdxl_params":       SDXL_PARAMS,
            "seed":              seed,
        }

    else:  # sd3
        run_sd3_edit = modal.Function.from_name("ultraedit-sd3-edit", "run_sd3_edit")
        if exp["sd3_prompt_mode"] == "instruction":
            edit_prompt = meta["instruction"]
        else:
            edit_prompt = meta["source_caption_long"] + " " + meta["instruction"]

        fut = run_sd3_edit.spawn(
            image_bytes    = sample["image_bytes"],
            mask_bytes     = mf_bytes,
            edit_prompt    = edit_prompt,
            source_prompt  = meta["source_caption_long"],
            seed           = seed,
            **SD3_PARAMS,
        )
        meta_used = {
            "experiment":       exp["name"],
            "description":      exp["description"],
            "sd3_prompt_mode":  exp["sd3_prompt_mode"],
            "edit_prompt":      edit_prompt,
            "prompt_token_est": len(edit_prompt.split()),
            "sd3_params":       SD3_PARAMS,
            "seed":             seed,
        }

    return fut, meta_used, t0


# ── Local entrypoint ───────────────────────────────────────────────────────────
app = modal.App("ultraedit-eval")

@app.local_entrypoint()
def main(
    experiments: str = "",   # comma-separated subset, e.g. "1_clip77,4_sd3_short"
    samples:     str = "",   # comma-separated subset, e.g. "sample_01,sample_03"
    seed:        int = 42,
):
    """
    modal run run_eval.py
    modal run run_eval.py --experiments "1_clip77,4_sd3_short"
    modal run run_eval.py --samples "sample_01"
    """
    from PIL import Image  # ensure PIL available locally

    # ── Filter experiments ─────────────────────────────────────────────────────
    exps = EXPERIMENTS
    if experiments:
        wanted = [e.strip() for e in experiments.split(",")]
        exps = [e for e in EXPERIMENTS if e["name"] in wanted]
        print(f"Running experiments: {[e['name'] for e in exps]}")

    # ── Load samples ───────────────────────────────────────────────────────────
    sample_dirs = sorted(EVAL_DATA.glob("sample_*/"))
    if samples:
        wanted_s = [s.strip() for s in samples.split(",")]
        sample_dirs = [s for s in sample_dirs if s.name in wanted_s]

    loaded = []
    for sd in sample_dirs:
        try:
            loaded.append(load_sample(sd))
        except FileNotFoundError as e:
            print(f"[WARN] {sd.name}: {e} — skipping")

    if not loaded:
        print("No valid samples found. Run run_eval_preprocess.py first.")
        return

    total = len(exps) * len(loaded)
    print(f"\n{'='*60}")
    print(f"  UltraEdit Eval — {len(exps)} experiments × {len(loaded)} samples")
    print(f"  Total jobs: {total}  (all dispatched in parallel)")
    print(f"{'='*60}")

    # ── Dispatch all jobs in parallel ─────────────────────────────────────────
    futures = []   # (exp_name, sample_id, fut, meta_used, t0)
    for exp in exps:
        for sample in loaded:
            sid = sample["meta"]["sample_id"]
            print(f"  → Spawning [{exp['name']}] [{sid}]...")
            try:
                fut, meta_used, t0 = dispatch_job(exp, sample, seed=seed)
                futures.append((exp["name"], sid, fut, meta_used, t0))
            except Exception as e:
                print(f"  [DISPATCH ERROR] {exp['name']} / {sid}: {e}")

    print(f"\n  All {len(futures)} jobs dispatched. Collecting results...\n")

    # ── Collect results ────────────────────────────────────────────────────────
    total_ok  = 0
    total_err = 0
    for ename, sid, fut, meta_used, t0 in futures:
        print(f"  Collecting [{ename}] [{sid}]...")
        try:
            result  = fut.get()
            elapsed = time.time() - t0
            save_result(ename, sid, result, meta_used, elapsed)
            total_ok += 1
        except Exception as e:
            print(f"  [ERROR] {ename} / {sid}: {e}")
            total_err += 1

    print(f"\n{'='*60}")
    print(f"  Done! {total_ok}/{total} OK, {total_err} errors")
    print(f"  Results saved to: eval_results/")
    print(f"{'='*60}\n")
