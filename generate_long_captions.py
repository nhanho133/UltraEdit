"""
generate_long_captions.py
=========================
Sinh long caption cho UltraEdit dataset (BleachNick/UltraEdit_w_mask) bằng InternVL2:

  Pipeline cho mỗi sample:
    source_image  ──InternVL2──▶  source_caption_long   (~150-250 words)
    source_caption_long
        + instruction (từ UltraEdit) ──InternVL2 text-only──▶  target_caption_long
                                                                (minimal change)

  Output: JSONL  { idx, instruction, source_caption_long, target_caption_long, similarity }

Hai entrypoints:
  A) run_eval   — xử lý eval_data/sample_*/ (nhỏ, dùng để test/debug)
  B) run_dataset — load thẳng từ HF cache BleachNick/UltraEdit_w_mask, xử lý hàng loạt

Usage:
  # Test trên eval samples
  modal run generate_long_captions.py::run_eval --samples "sample_03,sample_fh01"
  modal run generate_long_captions.py::run_eval --dry-run

  # Scale: xử lý 1000 samples đầu từ HF dataset
  modal run generate_long_captions.py::run_dataset --start 0 --end 1000

  # Xử lý shard lớn
  modal run generate_long_captions.py::run_dataset --start 0 --end 50000 --shard-size 500
"""

import modal
import os
from pathlib import Path

app = modal.App("ultraedit-caption-gen")

# ── Modal Image ────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("numpy<2")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.40.0",
        "accelerate>=0.24.0",
        "huggingface_hub>=0.23.0",
        "sentencepiece>=0.1.99",
        "datasets>=2.19.0",
        "einops",
        "timm",
        "Pillow>=9.0.0",
    )
)

model_cache = modal.Volume.from_name("ultraedit-model-cache", create_if_missing=True)
CACHE_DIR = "/cache"
HF_CACHE  = "/cache/huggingface"

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_CAPTION_PROMPT = (
    "Describe this image in detail. Include: the main subject(s) and their attributes, "
    "the setting/background, colors, lighting, spatial relationships, and any notable details. "
    "Write a single descriptive paragraph."
)

MODIFY_CAPTION_PROMPT = """\
You are a precise text editor. Minimally modify the caption based on the edit instruction.

Rules:
1. Only change the words/phrases directly related to the edit instruction.
2. Keep ALL other details exactly as they are.
3. Preserve the same sentence structure and length as much as possible.
4. Output ONLY the modified caption, nothing else.

Caption:
{caption}

Edit instruction:
{instruction}

Modified caption:"""

STRICT_MODIFY_PROMPT = """\
Rewrite the caption below by ONLY changing the part about: '{instruction}'.
Keep EVERYTHING else word-for-word unchanged.

Caption: {caption}

Rewritten caption:"""


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _clean_caption(modified: str, original: str) -> str:
    modified = modified.strip()
    for prefix in [
        "Modified caption:", "Modified Caption:", "Caption:", "Output:",
        "Here is", "Here's", "The modified", "Answer:", "Rewritten caption:",
    ]:
        if modified.lower().startswith(prefix.lower()):
            modified = modified[len(prefix):].strip().lstrip(":").strip()
    if len(modified.split()) < len(original.split()) * 0.4:
        return original
    return modified


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _build_internvl2(model_name: str, hf_cache: str):
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoTokenizer, AutoModel

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=hf_cache
    )
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=hf_cache,
    ).eval().cuda()
    return model, tokenizer, transform


def _dynamic_preprocess(image, min_num=1, max_num=6, image_size=448):
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h
    candidates = [
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    ] or [(1, 1)]
    best = min(candidates, key=lambda r: abs(aspect - r[0] / r[1]))
    tw, th = best[0] * image_size, best[1] * image_size
    resized = image.resize((tw, th))
    tiles = [
        resized.crop((c * image_size, r * image_size,
                      (c + 1) * image_size, (r + 1) * image_size))
        for r in range(best[1])
        for c in range(best[0])
    ]
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _caption_one(model, tokenizer, transform, pil_img, instruction, gen_cfg):
    """
    source_image + instruction → (source_caption_long, target_caption_long, similarity)
    """
    import torch

    # Step 1: ảnh → source caption
    tiles = _dynamic_preprocess(pil_img)
    pv    = torch.stack([transform(t) for t in tiles]).to(torch.bfloat16).cuda()
    src   = model.chat(tokenizer, pv, SOURCE_CAPTION_PROMPT, gen_cfg).strip()

    # Step 2: source caption + instruction → target caption (text-only, no image)
    tgt = model.chat(
        tokenizer, None,
        MODIFY_CAPTION_PROMPT.format(caption=src, instruction=instruction),
        gen_cfg,
    )
    tgt = _clean_caption(tgt, src)

    # Retry nếu thay đổi quá nhiều
    if _jaccard(src, tgt) < 0.50:
        tgt = model.chat(
            tokenizer, None,
            STRICT_MODIFY_PROMPT.format(caption=src, instruction=instruction),
            gen_cfg,
        )
        tgt = _clean_caption(tgt, src)

    return src, tgt, round(_jaccard(src, tgt), 4)


# ═════════════════════════════════════════════════════════════════════════════
# Remote function A — batch eval samples trong 1 container (load model 1 lần)
# ═════════════════════════════════════════════════════════════════════════════
@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: model_cache},
    timeout=1800,   # 30 min cho batch ~15 samples
    memory=20480,
)
def caption_eval_batch(
    samples: list,   # list of {"sid": str, "image_bytes": bytes, "instruction": str}
    model_name: str = "OpenGVLab/InternVL2-8B",
) -> list:           # list of {"sid", "source_caption_long", "target_caption_long", ...}
    import io
    from PIL import Image

    print(f"[InternVL2] Loading {model_name} ...")
    model, tokenizer, transform = _build_internvl2(model_name, HF_CACHE)
    gen_cfg = dict(max_new_tokens=400, do_sample=False, repetition_penalty=1.05)
    print(f"[InternVL2] Model loaded ✓  |  {len(samples)} samples to process")

    results = []
    for i, s in enumerate(samples):
        sid         = s["sid"]
        instruction = s["instruction"]
        try:
            pil_img = Image.open(io.BytesIO(s["image_bytes"])).convert("RGB")
            src, tgt, sim = _caption_one(
                model, tokenizer, transform, pil_img, instruction, gen_cfg
            )
            print(f"  [{i+1}/{len(samples)}] {sid}  sim={sim:.3f}  "
                  f"src={len(src.split())}w  tgt={len(tgt.split())}w")
            results.append({
                "sid":                 sid,
                "source_caption_long": src,
                "target_caption_long": tgt,
                "similarity":          sim,
                "source_word_count":   len(src.split()),
                "target_word_count":   len(tgt.split()),
            })
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] {sid}  ERROR: {e}")
            results.append({"sid": sid, "error": str(e)})

    ok = sum(1 for r in results if "error" not in r)
    print(f"[InternVL2] Done: {ok}/{len(samples)} success")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Remote function B — load từ HF UltraEdit dataset (streaming), xử lý 1 shard
# ═════════════════════════════════════════════════════════════════════════════
@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: model_cache},
    timeout=3600,
    memory=20480,
)
def process_ultraedit_shard(
    start_idx:    int,
    end_idx:      int,
    model_name:   str = "OpenGVLab/InternVL2-8B",
    dataset_name: str = "BleachNick/UltraEdit_w_mask",
    split:        str = "train",
) -> list:
    """
    Load BleachNick/UltraEdit_w_mask bằng streaming (không cần download toàn bộ).
    Xử lý items [start_idx, end_idx) — lấy instruction có sẵn trong dataset.

    Dataset fields:
      input_image  (PIL Image) — ảnh gốc
      edited_image (PIL Image) — ảnh đã edit
      instruction  (str)       — edit instruction
    """
    from datasets import load_dataset
    from PIL import Image

    n_items = end_idx - start_idx
    print(f"[Dataset] Streaming {dataset_name} split={split}  "
          f"range=[{start_idx}, {end_idx})  n={n_items}")

    # streaming=True: không download toàn bộ, chỉ đọc đúng range cần
    ds = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        cache_dir=HF_CACHE,
    )
    # Nhảy đến start_idx rồi lấy n_items
    shard_iter = ds.skip(start_idx).take(n_items)

    print(f"[InternVL2] Loading {model_name} ...")
    model, tokenizer, transform = _build_internvl2(model_name, HF_CACHE)
    gen_cfg = dict(max_new_tokens=400, do_sample=False, repetition_penalty=1.05)
    print("[InternVL2] Model loaded ✓")

    results = []
    for pos, item in enumerate(shard_iter):
        idx = start_idx + pos
        try:
            # ── Lấy instruction đã có sẵn trong UltraEdit ────────────────────
            instruction = (
                item.get("instruction")
                or item.get("edit_prompt")
                or ""
            ).strip()
            pil_img = item.get("input_image")

            if not instruction:
                print(f"  [SKIP] idx={idx}: missing instruction")
                results.append({"idx": idx, "error": "missing instruction"})
                continue
            if pil_img is None:
                print(f"  [SKIP] idx={idx}: missing input_image")
                results.append({"idx": idx, "error": "missing input_image"})
                continue

            if not isinstance(pil_img, Image.Image):
                pil_img = Image.fromarray(pil_img).convert("RGB")
            else:
                pil_img = pil_img.convert("RGB")

            src, tgt, sim = _caption_one(
                model, tokenizer, transform, pil_img, instruction, gen_cfg
            )
            results.append({
                "idx":                 idx,
                "instruction":         instruction,
                "source_caption_long": src,
                "target_caption_long": tgt,
                "similarity":          sim,
                "source_word_count":   len(src.split()),
                "target_word_count":   len(tgt.split()),
            })

            if (pos + 1) % 10 == 0 or pos == 0:
                print(f"  [{pos+1}/{n_items}] idx={idx} "
                      f"sim={sim:.3f}  src={len(src.split())}w  tgt={len(tgt.split())}w")

        except Exception as e:
            print(f"  [ERROR] idx={idx}: {e}")
            results.append({"idx": idx, "error": str(e)})

    ok = sum(1 for r in results if "error" not in r)
    print(f"[Shard] Done: {ok}/{n_items} success  range=[{start_idx},{end_idx})")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Local entrypoint A — eval_data/sample_*/
# ═════════════════════════════════════════════════════════════════════════════
@app.local_entrypoint()
def run_eval(
    samples:    str  = "",
    model_name: str  = "OpenGVLab/InternVL2-8B",
    dry_run:    bool = False,
    overwrite:  bool = False,
):
    """
    Sinh long caption cho eval_data/sample_*/ (image.png + instruction từ metadata.json).

      modal run generate_long_captions.py::run_eval
      modal run generate_long_captions.py::run_eval --samples "sample_03,sample_fh01"
      modal run generate_long_captions.py::run_eval --dry-run
      modal run generate_long_captions.py::run_eval --overwrite
    """
    import json

    EVAL_DATA = Path("eval_data")
    all_dirs  = sorted(EVAL_DATA.glob("sample_*/"))
    if samples:
        wanted   = {s.strip() for s in samples.split(",")}
        all_dirs = [d for d in all_dirs if d.name in wanted]

    print(f"\n{'='*60}")
    print(f"  InternVL2 Caption — eval samples")
    print(f"  Model   : {model_name}")
    print(f"  Samples : {len(all_dirs)}  dry_run={dry_run}  overwrite={overwrite}")
    print(f"{'='*60}\n")

    to_process = []
    for sd in all_dirs:
        meta_path = sd / "metadata.json"
        if not meta_path.exists() or not (sd / "image.png").exists():
            print(f"  [SKIP] {sd.name}: missing image.png or metadata.json")
            continue
        meta = json.loads(meta_path.read_text())
        # FineHARD samples dùng region-level captions từ bbox_info — KHÔNG dùng InternVL2
        if meta.get("finehard_id") and not overwrite:
            print(f"  [SKIP] {sd.name}: FineHARD sample (finehard_id={meta['finehard_id']}), "
                  f"captions are ground-truth region labels. Use --overwrite to force.")
            continue
        existing = meta.get("source_caption_long", "")
        # >50 words = đã có long caption thực sự (hand-crafted hoặc InternVL2 trước đó)
        if len(existing.split()) > 50 and not overwrite:
            print(f"  [SKIP] {sd.name}: already has long caption ({len(existing.split())}w). "
                  f"Use --overwrite to redo.")
            continue
        to_process.append((sd, meta, meta_path))

    if not to_process:
        print("  Nothing to process.")
        return

    print(f"  → {len(to_process)} sample(s) to process in 1 container...\n")

    # Gom tất cả vào 1 batch call duy nhất — load model 1 lần duy nhất
    batch_input = [
        {
            "sid":         sd.name,
            "image_bytes": (sd / "image.png").read_bytes(),
            "instruction": meta.get("instruction", ""),
        }
        for sd, meta, _ in to_process
    ]
    print(f"  Submitting 1 batch job to Modal ({len(batch_input)} samples)...\n")
    all_results = caption_eval_batch.remote(batch_input, model_name)

    # Index kết quả theo sid
    result_map = {r["sid"]: r for r in all_results}

    success = 0
    for sd, meta, meta_path in to_process:
        result = result_map.get(sd.name, {})
        print(f"\n[{sd.name}]")
        print(f"  instruction : {meta.get('instruction','')}")

        if "error" in result:
            print(f"  ✗ ERROR: {result['error']}")
            continue
        if not result:
            print(f"  ✗ No result returned")
            continue

        print(f"  source cap  : ({result['source_word_count']}w) "
              f"{result['source_caption_long'][:110]}...")
        print(f"  target cap  : ({result['target_word_count']}w) "
              f"{result['target_caption_long'][:110]}...")
        print(f"  similarity  : {result['similarity']:.3f}")

        if dry_run:
            print(f"  [DRY-RUN] skip write")
            continue

        meta["source_caption_long"] = result["source_caption_long"]
        meta["target_caption_long"] = result["target_caption_long"]
        meta["caption_meta"] = {
            "model":        model_name,
            "similarity":   result["similarity"],
            "source_words": result["source_word_count"],
            "target_words": result["target_word_count"],
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"  ✓ metadata.json updated → {meta_path}")
        success += 1

    print(f"\n{'='*60}  Done! {success}/{len(to_process)} updated.\n")


# ═════════════════════════════════════════════════════════════════════════════
# Local entrypoint B — HF UltraEdit dataset (scale)
# ═════════════════════════════════════════════════════════════════════════════
@app.local_entrypoint()
def run_dataset(
    start:        int  = 0,
    end:          int  = 1000,
    shard_size:   int  = 200,
    model_name:   str  = "OpenGVLab/InternVL2-8B",
    dataset_name: str  = "BleachNick/UltraEdit_w_mask",
    split:        str  = "train",
    output_file:  str  = "data/ultraedit_captions.jsonl",
    resume:       bool = True,
):
    """
    Load BleachNick/UltraEdit_w_mask (từ HF cache trên Modal volume — streaming).
    Instruction đã có sẵn trong dataset, dùng để modify target caption.

      # Test nhỏ
      modal run generate_long_captions.py::run_dataset --start 0 --end 10

      # Scale lớn hơn
      modal run generate_long_captions.py::run_dataset --start 0 --end 1000
      modal run generate_long_captions.py::run_dataset --start 0 --end 50000 --shard-size 500

      # Resume tự động (skip shards đã xong)
      modal run generate_long_captions.py::run_dataset --start 0 --end 1000

      # Chạy lại từ đầu
      modal run generate_long_captions.py::run_dataset --no-resume
    """
    import json

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: xác định các shard đã hoàn thành dựa vào idx đã ghi
    done_idx_set: set = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    done_idx_set.add(item["idx"])
                except Exception:
                    pass
        print(f"  Resume: {len(done_idx_set)} idx already in output, skipping completed shards.")

    # Tạo shard ranges [start, end) bước shard_size
    # Bỏ qua shard nếu toàn bộ idx trong shard đã được ghi rồi
    shard_ranges = []
    for s in range(start, end, shard_size):
        e = min(s + shard_size, end)
        shard_idx = set(range(s, e))
        if resume and shard_idx.issubset(done_idx_set):
            print(f"  [SKIP shard] [{s}, {e}): already done")
            continue
        shard_ranges.append((s, e))

    if not shard_ranges:
        print("  All shards already processed. Done!")
        return

    total_items = sum(e - s for s, e in shard_ranges)
    print(f"\n{'='*60}")
    print(f"  InternVL2 Caption — BleachNick/UltraEdit_w_mask (streaming)")
    print(f"  Dataset  : {dataset_name}  split={split}")
    print(f"  Range    : [{start}, {end})  total={end - start}")
    print(f"  Shards   : {len(shard_ranges)} × ~{shard_size}  ({total_items} items to run)")
    print(f"  Model    : {model_name}")
    print(f"  Output   : {output_file}")
    print(f"{'='*60}\n")

    # Mỗi tuple = (start_idx, end_idx, model_name, dataset_name, split)
    shard_inputs = [
        (s, e, model_name, dataset_name, split)
        for s, e in shard_ranges
    ]

    total_ok = total_err = 0
    with open(out_path, "a") as out_f:
        for (s, e), shard_result in zip(shard_ranges,
                                         process_ultraedit_shard.starmap(shard_inputs)):
            for item in shard_result:
                # Bỏ qua idx đã có trong file (edge case resume giữa shard)
                if resume and item["idx"] in done_idx_set and "error" not in item:
                    continue
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_f.flush()

            ok  = sum(1 for r in shard_result if "error" not in r)
            err = len(shard_result) - ok
            total_ok  += ok
            total_err += err

            sims    = [r["similarity"] for r in shard_result if "error" not in r]
            sim_avg = sum(sims) / len(sims) if sims else 0.0
            src_avg = (
                sum(r.get("source_word_count", 0) for r in shard_result if "error" not in r)
                / max(ok, 1)
            )
            print(f"  Shard [{s},{e}): ok={ok}  err={err}  "
                  f"sim_avg={sim_avg:.3f}  src_words_avg={src_avg:.0f}w")

    print(f"\n{'='*60}")
    print(f"  Total  : {total_ok + total_err}")
    print(f"  Success: {total_ok}")
    print(f"  Errors : {total_err}")
    print(f"  Output : {out_path.resolve()}")
    print(f"{'='*60}\n")
