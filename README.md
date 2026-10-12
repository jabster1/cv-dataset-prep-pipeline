# cv-dataset-pipeline

A local, reproducible pipeline that turns raw images and videos into a clean,
leak-free, YOLO-ready object-detection dataset. Drop media into a folder, list your class
names in a YAML config, and run one command.

Nothing is project-specific: the first use case is Lockout/Tagout (LOTO) detection, but
the same pipeline works for tools, industrial equipment, or anything else.

> **Status:** Phases 1–2 of 7 are implemented (foundation, ingestion, deduplication). See
> [Roadmap](#roadmap). Commands for later stages are documented as they land.

---

## Principles

- **Raw data is immutable.** The raw directory is only ever opened for reading; the config
  refuses a workspace that overlaps it.
- **Nothing is silently discarded.** Every raw file ends up in `items.csv` or in
  `rejected.csv` with a reason.
- **Lineage everywhere.** Every image records its source file, source hash, frame index /
  timestamp, and (for augmentations) its parent.
- **No leakage.** Splitting is grouped by source: all frames of a video, an image and its
  augmentations, and near-duplicates always land in the same split.
- **Deterministic.** Content-addressed IDs + fixed seeds → same inputs and config produce
  byte-identical manifests.
- **Local-first.** Plain files (CSV / JSON / YAML) — no database, no services.

## Setup

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]" -c constraints.txt
pytest
```

[`constraints.txt`](constraints.txt) pins the tested combination of the dependencies that
matter for reproducibility. It is especially important for OpenCV: Ultralytics depends on
`opencv-python` and albumentations on `opencv-python-headless`, and both install the same
`cv2` module, so their versions must match. Torch is left to pip and Ultralytics because
its wheels are platform-specific. Installing without `-c constraints.txt` works but may
resolve untested versions.

## Quick start (synthetic demo data)

```bash
python examples/make_synthetic_raw.py                       # writes data/example/raw
python -m cv_pipeline prepare --config configs/example.yaml
```

Then look in `workspaces/example/`.

## Commands

| Command | What it does | Status |
|---|---|---|
| `python -m cv_pipeline prepare --config C` | Run every preparation stage in order | ✅ (currently: ingest → dedup) |
| `python -m cv_pipeline ingest --config C` | Copy images, sample video frames, write lineage manifest | ✅ |
| `python -m cv_pipeline dedup --config C` | Exact + near-duplicate detection, split groups, duplicates report | ✅ |
| `python -m cv_pipeline show-config --config C` | Print the fully resolved config | ✅ |
| `annotate`, `split`, `augment`, `export` | Individual stages | Phases 3–6 |
| `python -m cv_pipeline validate --config C` | Dataset checks + report | Phase 6 |
| `python -m cv_pipeline train --config C` | Optional Ultralytics training | Phase 7 |

Common flags: `--force` recomputes outputs instead of reusing them; `--log-level DEBUG`
shows per-file detail.

## Adding images and videos

Put media under the configured `ingest.raw_dir` (scanned recursively — subfolders are fine):

```
data/loto/raw/
    images/   *.jpg *.jpeg *.png *.webp
    videos/   *.mp4 *.mov
```

- Images are copied byte-for-byte (EXIF preserved) to `interim/images/img_<sha16>.<ext>`.
- Videos are sampled into frames `interim/images/vid_<sha12>_f<frame>.jpg` according to
  `ingest.video` (`mode: fps` or `mode: every_n_frames`, optional `max_frames`, spread
  uniformly across the video by default).
- Re-running is incremental: unchanged files are reused. Renaming or moving a raw file does
  **not** change its ID (IDs are derived from file contents).
- Unsupported, unreadable, and byte-identical duplicate files are listed in
  `manifests/rejected.csv`. Hidden files (`.DS_Store`, …) are ignored and counted in the log.

## Duplicates

`dedup` runs after ingest and never deletes or moves anything. It distinguishes:

| Kind | How it is found | Default handling |
|---|---|---|
| Byte-identical raw files (incl. renamed/moved copies) | SHA-256 at ingest | Only the canonical copy is ingested; the rest are listed in `rejected.csv` and the report |
| Byte-identical items | same `sha256` | Non-canonical copies are **excluded** (`exact_duplicate_action`) |
| Pixel-identical items (same pixels, different bytes) | SHA-256 of decoded pixels | Non-canonical copies are **excluded** |
| Near-duplicates across sources (two photos of one scene, a photo and a video frame) | pHash ≤ `near_duplicate_threshold` **and** dHash ≤ `dhash_threshold` | **Kept**, linked, and put in the **same split group** |
| Near-duplicates within one source (adjacent video frames) | same | **Never linked or excluded**; summarised per video in the report |

Why same-source similarity is ignored: a video's frames already share a `source_id`, so
they always land in the same split and can't leak. A run of similar frames is often
exactly the data you want. If the report shows a video is highly redundant, lower
`ingest.video.fps` instead.

"Excluded" means `excluded=true` in `manifests/dedup.csv` with a reason and
`duplicate_of`. Later stages skip those items, but the files and their `items.csv` rows
remain.

**Which identical copy is canonical** (deterministic; lower wins, see
[`cv_pipeline/canonical.py`](cv_pipeline/canonical.py)):

1. A path that doesn't look like a copy. "Looks like a copy" means a directory or
   file-stem word is `duplicate(s)`, `dup(s)`, `copy`, `copies` or `backup`, or the name
   ends in ` (1)`.
2. A video frame before a still image. This only applies to pixel-identical items: a still
   identical to an extracted frame was exported from it, and keeping the frame keeps the
   sequence whole.
3. Fewer directory levels.
4. The path with Unicode normalised and case ignored, then the path as-is (then frame
   index and ID).

For example, `images/scene.jpg` beats `images/duplicates/scene.jpg`, `images/scene (1).jpg`
and `images/site/a/scene.jpg`.

Outputs:

- `manifests/dedup.csv`: per item, its hashes, `dup_group`, `excluded` / `exclude_reason` /
  `duplicate_of`, and `split_group` (the unit the splitter will assign to train/val/test)
- `manifests/duplicate_pairs.csv`: every linked pair with its kind and Hamming distances
- `reports/duplicates.json` + `reports/duplicates.md`: summary, groups, video redundancy,
  and warnings (e.g. one split group larger than the test split)
- `manifests/image_hashes.csv`: hash cache, so re-runs only hash new files

Tuning: distances are in bits out of `hash_size²` (64 by default). Raise the thresholds
to catch more aggressive edits (crops, re-framing); lower them if unrelated images get
linked. Pair search compares every pair, so cost grows with n²: about 0.6 s for 5k items
and 8.5 s for 20k on a laptop.

## Configuration

See [`configs/example.yaml`](configs/example.yaml) (fully commented) and
[`configs/loto.example.yaml`](configs/loto.example.yaml). Top-level sections:

| Section | Purpose |
|---|---|
| `root` | Base for all relative paths; itself relative to the config file (`..` = repo root) |
| `project` | `name`, `workspace` (output dir), `seed` |
| `classes` | Class names in YOLO id order |
| `ingest` | `raw_dir`, accepted extensions, video sampling |
| `dedup` | Hash size, pHash/dHash thresholds, exact/near-duplicate actions, grouping for splits |
| `annotation` | Backend (`yolo`), `model`, confidence / review thresholds, `class_map` |
| `split` | `train` / `val` / `test` fractions (must sum to 1) |
| `augmentation` | Copies per **train** image, transforms and their probabilities |
| `training` | Ultralytics model, epochs, imgsz, batch, device, patience |

Loading is strict: unknown keys, wrong types, and inconsistent values (e.g. split fractions
not summing to 1) fail immediately with a clear message.

## Architecture

The pipeline is a sequence of independent stages. Each reads the previous stage's
manifest from the workspace and writes its own, so stages can be run (and debugged)
individually, and `prepare` is just "run them in order".

```
raw/ ──ingest──▶ items.csv ──dedup──▶ duplicates.csv ──annotate──▶ labels/ + review/
                                                                        │ (human review)
dataset/ ◀──export── augmentations.csv ◀──augment (train only)── splits.csv ◀──split──┘
   │
   └──validate──▶ reports/validation.{json,md}            train ──▶ runs/   (separate)
```

```
cv_pipeline/
    cli.py           argparse entry point (python -m cv_pipeline)
    config.py        typed schema, strict YAML loading, validation, path resolution
    pipeline.py      stage registry used by `prepare`
    workspace.py     workspace directory layout
    manifest.py      typed CSV manifests (ItemRecord, RejectedRecord, DedupRecord, ...)
    synthetic.py     synthetic scenes/videos for tests and demos
    ingest/          image copy + video frame sampling
    dedup/           perceptual/pixel hashing, pair search, grouping, duplicates report
    utils/           structured logging, hashing, atomic file I/O
configs/             example.yaml, loto.example.yaml
examples/            make_synthetic_raw.py
tests/
```

### Workspace layout

```
<workspace>/
    config.resolved.yaml      exact config used by the last run
    interim/images/           ingested images + extracted frames (named by item_id)
    manifests/
        items.csv             one row per image, with lineage
        rejected.csv          files not carried forward, with reason
        dedup.csv             per-item duplicate status + split_group
        duplicate_pairs.csv   linked pairs with kind and distances
        image_hashes.csv      hash cache
    reports/
        ingest_summary.json
        duplicates.json / duplicates.md
    logs/pipeline.jsonl       structured JSON-lines log of every run
```

Later phases add `labels/`, `review/`, `dataset/`, `runs/` and more manifests.

### Lineage fields (`items.csv`)

| Column | Meaning |
|---|---|
| `item_id` | Stable content-addressed ID (`img_…`, `vid_…_f000123`, later `aug_…`) |
| `path` | File inside the workspace |
| `source_id` | **Grouping key for splits.** Same for every frame of a video / derivative of an image |
| `source_type` | `image`, `video_frame` (later `augmented`) |
| `source_path` | Original file, relative to `raw_dir` |
| `source_sha256` / `sha256` | Hash of the raw source / of this file |
| `width`, `height` | Pixel size (after EXIF orientation) |
| `frame_index`, `timestamp_s` | For video frames |
| `parent_item_id` | For augmented images |

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundation, config, logging, manifests, ingestion (images + videos), CLI, tests | ✅ |
| 2 | Exact + perceptual-hash near-duplicate detection, duplicates report | ✅ |
| 3 | Pluggable annotation (YOLO backend), previews, review queue + manifest, label import/export | ⏳ |
| 4 | Source-grouped deterministic train/val/test split, persistent split manifest | ⏳ |
| 5 | Albumentations augmentation of train only, bbox-safe, lineage | ⏳ |
| 6 | YOLO dataset export + `dataset.yaml`, validation checks, JSON + Markdown report | ⏳ |
| 7 | Optional `train` command, end-to-end test, full documentation | ⏳ |

## Testing

```bash
pytest
```

Tests generate a tiny synthetic dataset (images in jpg/png/webp, videos in mp4/mov) in a
temp directory — no real data needed.
