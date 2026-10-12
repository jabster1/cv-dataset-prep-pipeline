# cv-dataset-pipeline

A local, reproducible pipeline that turns raw images and videos into a clean,
leak-free, YOLO-ready object-detection dataset. Drop media into a folder, list your class
names in a YAML config, and run one command.

Nothing is project-specific: the first use case is Lockout/Tagout (LOTO) detection, but
the same pipeline works for tools, industrial equipment, or anything else.

> **Status:** Phase 1 of 7 is implemented (foundation + ingestion). See
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
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Quick start (synthetic demo data)

```bash
python examples/make_synthetic_raw.py                       # writes data/example/raw
python -m cv_pipeline prepare --config configs/example.yaml
```

Then look in `workspaces/example/`.

## Commands

| Command | What it does | Status |
|---|---|---|
| `python -m cv_pipeline prepare --config C` | Run every preparation stage in order | ✅ (currently: ingest) |
| `python -m cv_pipeline ingest --config C` | Copy images, sample video frames, write lineage manifest | ✅ |
| `python -m cv_pipeline show-config --config C` | Print the fully resolved config | ✅ |
| `dedup`, `annotate`, `split`, `augment`, `export` | Individual stages | Phases 2–6 |
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

## Configuration

See [`configs/example.yaml`](configs/example.yaml) (fully commented) and
[`configs/loto.example.yaml`](configs/loto.example.yaml). Top-level sections:

| Section | Purpose |
|---|---|
| `root` | Base for all relative paths; itself relative to the config file (`..` = repo root) |
| `project` | `name`, `workspace` (output dir), `seed` |
| `classes` | Class names in YOLO id order |
| `ingest` | `raw_dir`, accepted extensions, video sampling |
| `dedup` | Perceptual-hash size, near-duplicate threshold, grouping for splits |
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
    manifest.py      typed CSV manifests (ItemRecord, RejectedRecord)
    synthetic.py     synthetic scenes/videos for tests and demos
    ingest/          image copy + video frame sampling
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
    reports/ingest_summary.json
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
| 2 | Exact + perceptual-hash near-duplicate detection, duplicates report | ⏳ |
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
