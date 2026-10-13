# cv-dataset-pipeline

A local, reproducible pipeline that turns raw images and videos into a clean,
leak-free, YOLO-ready object-detection dataset. Drop media into a folder, list your class
names in a YAML config, and run one command.

Nothing is project-specific: the first use case is Lockout/Tagout (LOTO) detection, but
the same pipeline works for tools, industrial equipment, or anything else.

> **Status:** Phases 1–5 of 7 are implemented (foundation, ingestion, deduplication,
> automatic annotation, source-aware splitting, legacy dataset import and label
> provenance, train-only augmentation). See
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
| `python -m cv_pipeline prepare --config C` | Run every enabled preparation stage in order | ✅ (currently: ingest → dedup → annotate if `annotation.enabled` → split → augment if `augmentation.enabled`) |
| `python -m cv_pipeline ingest --config C` | Copy images, sample video frames, write lineage manifest | ✅ |
| `python -m cv_pipeline dedup --config C` | Exact + near-duplicate detection, split groups, duplicates report | ✅ |
| `python -m cv_pipeline annotate --config C [--limit N]` | Pre-label eligible images with a detector; build the review queue | ✅ |
| `python -m cv_pipeline split --config C [--dry-run]` | Source-aware train/val/test assignment that preserves history | ✅ |
| `python -m cv_pipeline split --config C --resplit --reason TEXT [--confirm TOKEN]` | Deliberately rebuild all assignments (previews unless confirmed) | ✅ |
| `python -m cv_pipeline import-dataset --config C --name N [--dry-run]` | Explicitly import an existing labeled YOLO/Roboflow dataset listed under `imports` | ✅ |
| `python -m cv_pipeline labels --config C [--allow-machine-unreviewed] [--allow-imported-unverified] [--allow-machine-empty]` | Resolve each item's label with provenance (trusted human labels only by default) | ✅ |
| `python -m cv_pipeline show-config --config C` | Print the fully resolved config | ✅ |
| `python -m cv_pipeline augment --config C [--dry-run]` | Augment trusted-labelled **train** items only | ✅ |
| `export` | YOLO dataset export | Phase 6 |
| `python -m cv_pipeline validate --config C` | Dataset checks + report | Phase 6 |
| `python -m cv_pipeline train --config C` | Optional Ultralytics training | Phase 7 |

Common flags: `--force` recomputes outputs instead of reusing them; `--log-level DEBUG`
shows per-file detail.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | A stage could not run (missing inputs, missing or invalid model, unknown backend, ...) |
| 2 | Invalid configuration |
| 3 | Annotation completed and all outputs were written, but one or more items failed inference (see `reports/annotation.md`) |
| 4 | Split-history conflict. Nothing in `splits/` was changed (see `reports/split.md`) |

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

## Automatic annotation

`annotate` pre-labels images with an existing detector so people only have to correct
predictions instead of drawing every box. **Predictions are never treated as ground
truth.** Every box has `provenance=machine`. An item is either `needs_review` or
`unreviewed`, never "verified". Human review is a separate later step that records its own
provenance.

```
dedup-eligible item ─▶ backend ─▶ RawDetection ─▶ confidence/review policy ─▶ candidates + review status
                                                                          ─▶ immutable run snapshot ─▶ review queue
```

- **Eligible items:** items in `dedup.csv` that are not excluded, in `item_id` order.
  Excluded duplicates are never sent to the model. `--limit N` annotates only the first N.
- **Backends:** `backend: yolo` uses a local Ultralytics `.pt` detector. The model path is
  only checked when annotation actually runs, and **models are never downloaded**. Other
  detectors (an open-vocabulary model, ONNX, ...) plug in as one module that implements
  `AnnotationBackend` and calls `register_backend("name")`. Backend-specific settings go
  in `annotation.options`.
- **Class mapping:** a model label maps to a project class through `class_map`, or
  implicitly when the names are equal. Annotation refuses to start if no model class maps to
  a project class.

### Confidence and review policy

Per box (first matching rule wins), all recorded in `predictions.csv`:

| Rule | Decision |
|---|---|
| Not finite, or zero area after clipping to the image | `discarded_invalid_box` |
| Label maps to no project class | `discarded_unmapped_class` |
| `confidence < confidence_threshold` | `discarded_below_min_confidence` |
| Otherwise | `candidate`; `review_flag` if `confidence < review_threshold` |

An item goes to review (`needs_review`) for any of these reasons:

- `inference_failed` or `unreadable_image`
- `low_confidence`: any flagged candidate
- `no_detections`: `review_empty`
- `unmapped_class`: `review_unmapped_classes`
- `audit_sample`: a deterministic `review_sample_rate` (default 5%) sample of otherwise
  unflagged items, so confident predictions are still spot-checked

### Run snapshots and identity

Each completed run is an **immutable snapshot** in `annotations/runs/<run_id>/`. A new run
is built in a temporary directory and renamed into place. An existing run directory is
verified, never rewritten. `annotations/active_run.json` points to the current run.

| Fingerprint | Built from |
|---|---|
| `inference_fp` | Backend name and version, torch version, `model_sha256`, imgsz, iou, confidence floor, max detections, batch size, device, `options` |
| `policy_fp` | Classes, `class_map`, thresholds, review settings, seed |
| `snapshot_fp` | Ordered eligible items: `item_id`, lineage, `split_group` at the time of the run, `file_sha256`, `input_pixel_sha256`, size |
| `result_digest` | Ordered per-item status, error and normalised raw detections |
| `run_id` | `ann_` + sha256(spec = inference + policy + model reference, snapshot, result)[:16] |

| Change | Effect |
|---|---|
| Nothing | Same `run_id`; nothing written; the model isn't even loaded |
| New items | New run; only the new images are inferred; the old run is untouched |
| Thresholds or `class_map` | New run; no inference (cache reused) |
| Dedup `split_group` of an eligible item | New run; no inference (cache reused); the old run keeps its grouping |
| Model or inference settings | New run with fresh inference |
| `--force` | Re-infers everything. If the results are identical, the same run is reported as reproduced. If not, a new run is created and the changed items are listed. |
| Failed items | Never cached, so they are retried on the next run |

The **inference cache** (`annotations/cache/inference/`) is keyed by `inference_fp` and
`input_pixel_sha256`, the identity of the exact decoded array handed to the backend, not
the file bytes. If a decoder change alters the pixels, the cache misses instead of serving
stale predictions. `input_pixel_sha256` is
`sha256(b"cvp-input-pixels-v1\n" + f"{H} {W} 3 uint8 BGR HWC C\n" + C-contiguous pixel bytes)`
over the `uint8` BGR result of `cv2.imdecode(..., IMREAD_COLOR)`, with EXIF orientation
applied. The backend's own preprocessing (letterboxing, RGB conversion, scaling) happens
after this point and is covered by `inference_fp`.

Each run's `items.csv` records the `split_group` each item had when the run was created, so
later stages never need to reconstruct historical grouping from a `dedup.csv` that may
have changed since.

**Model reference.** Run metadata stores the model as a machine-independent reference, not
an absolute path:

- A file inside the project `root` is stored root-relative, e.g. `models/loto.pt`. This is
  the same whether the config wrote it as a relative or an absolute path.
- A file outside the project is stored as `external:<file name>`.

Moving the repository or model to another machine or home directory therefore doesn't
change the run. What the model actually *is* comes from `model_sha256`.

Hashes kept apart:

| Hash | What it identifies |
|---|---|
| `source_sha256` | The raw source file |
| `file_sha256` | The encoded workspace image file |
| `pixel_sha256` | Decoded pixels, used by dedup only |
| `input_pixel_sha256` | The backend input |
| `model_sha256` | The weights file |

## Reviewing predictions

`review/` is written in a layout YOLO-capable labeling tools (CVAT, Label Studio,
Roboflow, labelImg) can open:

```
review/
    images/<item_id>.<ext>    copies of queued images
    labels/<item_id>.txt      machine labels as a starting point (YOLO format)
    previews/<item_id>.jpg    green = candidate, amber = low confidence, grey = discarded
    classes.txt               class names in id order
    review_manifest.csv       item, source, predicted classes, min/max confidence, reasons,
                              based_on_run_id, label_state
```

Edit labels in place. **Human edits are never overwritten:** a label file is rewritten
only if it still holds exactly what the pipeline last wrote (tracked in
`review/.export_state.csv`). Otherwise it is kept and marked `label_state=edited`. Nothing
under `review/` is ever deleted.

Reviewing makes a machine label trusted only through an explicit entry in
`review/decisions.csv` (see "Label provenance"). An edited file alone is never
interpreted as a review.

## Importing an existing labeled dataset

Projects often start with a historical, manually annotated dataset, for example a
Roboflow YOLO export that trained an earlier model. Import it explicitly. `prepare` never
crawls for datasets.

```yaml
imports:
  - name: loto_roboflow_v1          # immutable once imported; a new version gets a new name
    path: data/loto/legacy/roboflow_v1
    format: yolo
    labels_trusted: true            # made by people -> provenance human_imported
    empty_labels_are_negatives: true  # empty label files = "reviewed, no target objects"
    ignored_only_labels_are_negatives: false  # see "Empty labels and negative images"
    class_map: {lock: padlock}      # source class -> project class
    ignore_classes: []              # source classes deliberately dropped
```

```bash
python -m cv_pipeline import-dataset --config C --name loto_roboflow_v1 --dry-run   # validate only
python -m cv_pipeline import-dataset --config C --name loto_roboflow_v1
python -m cv_pipeline prepare --config C        # imported + new data: ingest -> dedup -> ... -> split
python -m cv_pipeline labels --config C         # resolved labels + provenance report
```

- **Supported layouts:** Roboflow (`train|valid|test/images|labels`) and Ultralytics
  (`images|labels/train|val|test`), each with a `data.yaml` whose `names` may be a list or
  an id → name mapping.
- **The export is only read.** A verified byte copy goes to `<workspace>/imports/<name>/`,
  together with `images.csv`, `boxes.csv`, `rejected.csv`, `import.json` and
  `reports/import_<name>.md`.
- **Labels are validated, never silently trusted.** Each line needs exactly 5 values, a
  valid class ID, finite coordinates with the centre in [0, 1] and width/height in (0, 1],
  and a box that stays inside the image. Any bad line marks the file `invalid`; the image
  is still imported, but unlabeled. Empty and missing label files are recorded explicitly.
- **The taxonomy must fit the project.** Every source class must be a project class,
  mapped in `class_map`, or listed in `ignore_classes`; otherwise the import fails before
  writing anything. Duplicate names and several source classes mapping to one project
  class are rejected.
- **Imports are part of the one corpus.** Ingest treats every configured import as another
  root (origin `import:<name>`), so imported images get normal content-addressed IDs and go
  through dedup, split groups and split history with the raw data.
- **Byte-identical files from different origins are one item.**
  `manifests/source_refs.csv` keeps every reference (origin, path, legacy split), and raw
  copies are preferred as canonical. Pixel-identical copies are excluded by dedup, and
  their labels transfer to the kept copy. Near-duplicates share a split group.
- **The legacy split is metadata only.** It is recorded in `source_refs.csv` and in the
  labels report's legacy-vs-current table, and never used as the pipeline split. Imported
  and new data are split fresh by group, together. Keeping an old test set would have to be
  a separate, explicit option (not implemented).
- **Integrity is checked twice.** Imported files are re-verified against `images.csv` at
  ingest, and workspace files are re-hashed before label output is produced.
- **Roboflow derivative families always stay together.** Roboflow renames exported files
  `<original stem>_<ext>.rf.<32 hex>.<ext>`, and an augmented export can contain several
  transformed versions of one original. Flips, crops and similar changes are often not
  caught by perceptual hashing.
  - Files in one imported dataset that share an original stem form a lineage family. The
    key is namespaced by dataset, `import:<name>:roboflow:<stem>`, so datasets never
    collide.
  - Dedup forces every family into one split group, even with dedup disabled.
  - A family is not a duplicate: nothing is excluded or deleted because of it.
  - Only names matching the pattern exactly form families; nothing is inferred otherwise.
  - Family keys are recorded in `images.csv` and `source_refs.csv`, and counted in the
    import and duplicates reports.
- **Recommended workflow:** export future Roboflow versions **without augmentation**, and
  let the pipeline own train-only augmentation. Older augmented exports remain safe
  because of the family grouping.
- **EXIF orientation is reported, not changed.** Imported images whose EXIF orientation tag
  isn't 1 are listed in the import report and log, because they decode rotated. Files are
  never modified.

## Label provenance and trust

Every label has explicit provenance:

| Provenance | From | Trusted by default |
|---|---|---|
| `human_reviewed_corrected` | `review/decisions.csv` `corrected`, plus the edited `review/labels/<item>.txt` | ✅ |
| `human_reviewed_accepted` | `review/decisions.csv` `accepted`, plus that run's machine labels | ✅ |
| `human_imported` | An import with `labels_trusted: true` | ✅ |
| `imported_unverified` | An import with `labels_trusted: false` | `--allow-imported-unverified` |
| `machine_unreviewed` | The active annotation run, only if it matches the current dataset state | `--allow-machine-unreviewed` |

`cv_pipeline.labels.LabelResolver` is the single interface later stages use. It returns
`trusted`, `opt_in`, `unavailable` (with a reason) or `conflict` for each item, so they
don't need to know where a label came from.

- The highest-priority usable label wins, in the order of the table above.
- Labels from exact-duplicate copies are included.
- Different labels at the same priority are a `conflict`, and no label is used.

**Empty labels and negative images:**

| Situation | Result |
|---|---|
| Imported empty file, trusted import, `empty_labels_are_negatives` | Trusted negative |
| Imported empty file otherwise | Unavailable (`empty_label_not_confirmed_negative`) |
| Imported file whose boxes are **all** in `ignore_classes`, trusted import, `ignored_only_labels_are_negatives: true` | Trusted negative, noted `ignored_only_source_annotation` |
| Imported ignored-only file otherwise (the default) | Unavailable (`ignored_only_unconfirmed_negative`). Dropping other classes doesn't prove the project's targets were looked for. |
| A human accepts a machine result with zero detections | Trusted negative |
| Machine zero detections | **Never** a negative, even with `--allow-machine-unreviewed`, unless `--allow-machine-empty` is also given |
| Failed or unreadable inference | No label |
| Not reviewed | Unavailable (`machine_label_not_reviewed`) |

### Review decisions (`review/decisions.csv`)

| Column | Required | Meaning |
|---|---|---|
| `item_id` | ✅ | |
| `based_on_run_id` | ✅ | The annotation run that was reviewed |
| `decision` | ✅ | `accepted` \| `corrected` \| `rejected` |
| `reviewer`, `notes` | | |
| `label_sha256` | | For `corrected`: if given, it must match the edited file |

- **`accepted`:** the machine label is correct as-is. It is invalid if the review label was
  edited; that should be `corrected`.
- **`corrected`:** the edited `review/labels/<item>.txt` is the trusted label.
- **`rejected`:** that run's machine label is never used, under any policy.
- The last row for an item wins.
- Rows are checked: unknown items or runs, items missing from the run, and image content
  that changed since the run are all reported as invalid and ignored.

`labels` writes `labels/resolved.csv`, `labels/boxes.csv` and `reports/labels.{json,md}`.
The report covers status, provenance, unavailable reasons, negatives, label status per
split, the legacy-vs-current split table, conflicts and invalid decisions.

## Train/val/test splitting

`split` assigns every eligible item to `train`, `val` or `test`. It depends only on the
validated ingest and dedup state (`item_id`, `source_id`, `split_group`, file identity),
**never on machine predictions**, so it works with annotation disabled, with manual
labels, or with no labels yet. `prepare` runs it after dedup (and after annotate, when
that's enabled).

Priorities, in order:

1. **Prevent leakage.** A dedup `split_group` is indivisible. All frames of a video, and
   all near- or exact-duplicates across sources, stay in one split.
2. **Preserve history.** A group that has been assigned keeps its split forever. Adding
   data never reshuffles existing val or test items.
3. **Match the requested fractions** as closely as whole groups allow. The report shows
   requested and achieved fractions and explains any gap.

**Input check.** Before splitting, the stage confirms that:

- `dedup.csv` covers exactly the items in `items.csv`, with the same `source_id`s;
- it was produced with the current dedup settings;
- it was computed from the files currently on disk (via the dedup hash cache).

If any check fails, the stage refuses to run (exit 1) and asks you to re-run dedup.

The validated rows (`item_id, source_id, source_type, frame_index, split_group,
file_sha256`) form the split-input snapshot, and its fingerprint `snapshot_fp` is recorded
with every revision. If the active annotation run was made from exactly this state, its
`run_id` is recorded as optional lineage. Otherwise lineage is reported as `stale` or
`none`, and splitting still proceeds.

### What happens to history

| Situation | Result |
|---|---|
| Same group (same set of sources) | Keeps its split (`retained`; `items_changed` if, e.g., a video was re-sampled) |
| New sources join a group whose sources were all in one split | The whole group **inherits** that split (`extended`) |
| A historical group divides | Each part keeps the split (`regrouped`) |
| A group disappears | Kept as `inactive` history; **restored** to its old split if it reappears |
| A new group | Allocated (below) |
| **A group now contains sources from different historical splits** | **Conflict**: exit 4, nothing in `splits/` is changed, `reports/split.md` explains it |

**Resolving a conflict.** A targeted entry in `split.conflict_resolutions` may only name
the **most protected** historical split (`test > val > train`):

| Historical splits | Allowed resolution |
|---|---|
| train + test | `test` |
| train + val | `val` |
| val + test | `test` |

Any other value is rejected. An entry for a group that isn't in conflict, and wasn't
resolved that way before, is a configuration error (exit 2). Moving held-out data toward
training requires a full resplit.

### Allocation of new groups

1. New groups are placed largest first, with a seeded, stable tie-break
   (`split.seed`, default `project.seed`).
2. Each group goes to the split with the smallest increase in target-relative squared
   error, `((c+n-t)² - (c-t)²) / t`. Ties prefer train, then val, then test.
3. Splits with fraction 0 never receive groups.
4. With `require_nonempty: true` (the default): if best-fit would leave a requested val or
   test split empty, the smallest new groups are reserved for those splits first, as long
   as a group remains for train.

History is never moved to fill a split. When filling is impossible (too few groups, or no
new groups), the report warns instead.

### Full resplit (deliberate only)

```bash
python -m cv_pipeline split --config C --resplit --reason "dedup thresholds re-tuned"
#   -> preview: every move (e.g. test->train: 12 items) and a confirmation token; nothing written
python -m cv_pipeline split --config C --resplit --reason "dedup thresholds re-tuned" --confirm <token>
```

- `prepare` never resplits.
- The token is tied to the current history, so it stops working as soon as anything
  changes.
- Before rewriting, the current history is archived to `splits/archive/rev_NNNN/`.
- Moves of held-out data toward training are flagged with a loud warning.

`--dry-run` computes and reports without writing anything.

### Outputs

| File | Role |
|---|---|
| `splits/groups.csv` | **Persistent history**, one row per group ever assigned: split, active/inactive status, source/item counts with set fingerprints, reason (`allocated`, `inherited`, `conflict_resolution`, `resplit`), first and last revision |
| `splits/group_sources.csv` | **Persistent** source-level memory: `source_id → split_group, split, status`. This is what detects merges and restores reappearing data. |
| `splits/revisions.jsonl` | **Append-only**, one line per revision that changed anything: mode, `snapshot_fp`, dedup settings fingerprint, annotation lineage, split config, change counts, applied resolutions, resplit details, and a `state_digest` that is checked on every load |
| `splits/items.csv` | **Current materialised view**: `item_id, source_id, source_type, split_group, split, reason, assigned_in_revision` |
| `reports/split.json` / `.md` | Requested vs achieved, why they differ, changes, warnings, conflicts |

The outputs are deterministic: the same snapshot, history and config give byte-identical
files, whatever the row or filesystem order.

## Train-only augmentation

`augment` creates augmented copies of **train** items whose label resolves under the
chosen policy. Val and test items are never augmented. Every augmented image keeps full
lineage to its parent.

```
validated dataset state + latest split revision + LabelResolver(policy)
   │  the split snapshot must equal the label/dataset snapshot, or nothing is written
   ▼
train items ──usable label?──▶ eligible ──▶ variants (seeded; retried) ──▶ augmentation/runs/<run_id>/
           └──────────────────▶ skipped (reported, never treated as background)
```

**Label policy (`augmentation.label_policy`):**

- `trusted_only` (default) uses only `human_imported`, `human_reviewed_accepted` and
  `human_reviewed_corrected` labels.
- `allow_machine_unreviewed` is a development/POC opt-in for unreviewed machine labels. It
  needs the annotation run to be current for this dataset state. Machine zero-detection
  results are still not negatives unless `allow_machine_empty: true` is also set.
- Each augmented item records the provenance of its parent's label. Machine labels are
  never promoted to trusted.

**Skipped train items** are reported by category: `label_unavailable`, `label_untrusted`
(not reviewed, rejected, unverified import, unconfirmed empty or ignored-only),
`label_conflict` and `annotation_failed`. A trusted human negative is valid training data
and is augmented box-free.

**Refusals (exit 1, nothing written):**

- no split yet;
- the split was made from a different dataset state (re-run `split`, and `ingest`/`dedup`
  first if data or imports changed);
- machine opt-in with a stale annotation run (re-run `annotate`);
- the shared dataset-state check fails;
- a workspace image differs from `items.csv`. Every file is re-hashed before generation,
  and each parent again while it's read.

**Bounding-box safety.** Boxes use Albumentations' YOLO format with `clip=True`,
`min_visibility` and `min_box_area`, and are then checked to be finite, non-empty and
inside the image.

- A positive image that loses **any** box is retried. This is configurable with
  `allow_partial_box_loss`; losing **all** boxes is never accepted.
- Retries use deterministic seeds, up to `max_attempts`. A variant that fails every attempt
  is recorded in `rejected.csv`; nothing silently becomes a negative.
- Trusted negatives stay box-free.

**Conservative default transforms** (generic; override per project under
`augmentation.transforms.<name>`; always applied in this fixed order):

| On by default | Off by default (opt in deliberately) |
|---|---|
| `brightness_contrast` (±0.15) | `horizontal_flip`: mirrors text, left/right, handle position |
| `hue_saturation_value` (hue **0**, sat ±15, val ±10) | `rotate` (±5°): boxes grow; orientation may carry meaning |
| `gaussian_blur` (3 px) | `gamma` |
| `gauss_noise` (std 0.01–0.03) | `perspective` |
| `image_compression` (quality 70–95) | `bbox_safe_crop` |
| `affine` (scale 0.9–1.1, translate ±5%, rotation 0) | |

Hue is 0 by default because colour can encode state, such as red/green indicators or
coloured tags. Disable anything that could change what an object means in your project.

**Run identity and immutability.** `run_id` covers:

- the dataset `snapshot_fp`;
- the split revision and its state digest;
- a digest of every train item's resolved label;
- the full resolved augmentation config (policy, variants, seed, box settings, retries,
  every transform and its parameters, output format);
- the versions of albumentations, albucore, opencv-python and numpy.

The same inputs give byte-identical images and manifests. Changing anything creates a new
run, and completed runs are verified and reused, never rewritten.

**Outputs.** `augmentation/runs/<run_id>/` contains:

- `images/` and `labels/` (YOLO);
- `items.csv`: `aug_item_id`, parent item, `source_id`, `split_group`, `split=train`, split
  revision, origin, Roboflow family key, label provenance, source ref and sha, parent
  negative, seed, attempt, recipe (applied transforms and sampled parameters), input and
  output sha256, boxes before and after;
- `skipped.csv`, `rejected.csv` and `run.json`.

`augmentation/active_run.json` points to the current run.
`reports/augmentation.{json,md}` includes a **train-only check** (augmented parents per
split; val and test must be 0), skip reasons, provenance, trusted negatives, machine labels
used, transform counts, rejections, and training items by source (raw, historical import,
Roboflow family members, generated). Augmented items never enter `items.csv`, dedup or
split history.

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
| `annotation` | Backend (`yolo`), `model`, thresholds, batch size, `class_map`, review triggers, audit sample rate, backend `options` |
| `review` | Which items to export (`flagged` / `all`), copy images, draw previews |
| `imports` | Existing labeled datasets to import explicitly: name, path, format, trust, negatives, class map, ignored classes |
| `split` | `train` / `val` / `test` fractions (sum to 1; val/test may be 0), tie-break `seed`, `require_nonempty`, `conflict_resolutions` |
| `augmentation` | Train-only: label policy, variants per item, seed, box safety (`min_visibility`, `min_box_area`, `allow_partial_box_loss`, `max_attempts`), output format, transform overrides |
| `training` | Ultralytics model, epochs, imgsz, batch, device, patience |

Loading is strict: unknown keys, wrong types, and inconsistent values (e.g. split fractions
not summing to 1) fail immediately with a clear message.

## Architecture

The pipeline is a sequence of independent stages. Each reads the previous stage's
manifest from the workspace and writes its own, so stages can be run (and debugged)
individually, and `prepare` is just "run them in order".

```
raw/ ──ingest──▶ items.csv ──dedup──▶ dedup.csv ──annotate──▶ annotations/ + review/
                                                                       │ (human review)
dataset/ ◀──export── augmentations.csv ◀──augment (train only)── splits/ ◀──split──┘
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
    annotation/      backend protocol + registry, YOLO backend, policy, identities, cache,
                     immutable run snapshots, review queue, previews, report
    dataset_state.py validated ingest/dedup state + split-input snapshot fingerprint
    imports/         YOLO/Roboflow export validation, immutable import snapshots
    labels/          provenance, review decisions, LabelResolver, labels report
    augmentation/    transform registry, train-only augmentation runs, report
    splitting/       history, reconciliation, deterministic allocation, split report
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
        source_refs.csv       every file reference per source (origin, path, legacy split)
    imports/<name>/           immutable import snapshots: files/, images.csv, boxes.csv,
                              rejected.csv, import.json
    labels/                   resolved.csv, boxes.csv (label resolution output)
    augmentation/
        active_run.json       pointer to the current augmentation run
        runs/<run_id>/        immutable: images/, labels/, items.csv, skipped.csv,
                              rejected.csv, run.json
    annotations/
        active_run.json       pointer to the current run (mutable)
        runs/<run_id>/        immutable: run.json, items.csv, predictions.csv, labels/
        cache/inference/      raw detections by (inference_fp, input_pixel_sha256)
        cache/models/         model class names by model sha256
    review/                   review queue (see "Reviewing predictions")
    splits/
        groups.csv            persistent group assignments (active + inactive)
        group_sources.csv     persistent source-level assignment memory
        revisions.jsonl       append-only revision log
        items.csv             current item -> split view
        archive/rev_NNNN/     history before each deliberate resplit
    reports/
        ingest_summary.json
        duplicates.json / duplicates.md
        annotation.json / annotation.md
        split.json / split.md
        import_<name>.json / .md
        labels.json / labels.md
        augmentation.json / augmentation.md
    logs/pipeline.jsonl       structured JSON-lines log of every run
```

Later phases add `dataset/`, `runs/` and more manifests.

### Lineage fields (`items.csv`)

| Column | Meaning |
|---|---|
| `item_id` | Stable content-addressed ID (`img_…`, `vid_…_f000123`, later `aug_…`) |
| `path` | File inside the workspace |
| `source_id` | **Grouping key for splits.** Same for every frame of a video / derivative of an image |
| `source_type` | `image`, `video_frame` (later `augmented`) |
| `origin` | `raw` or `import:<name>` |
| `source_path` | Original file, relative to its origin's root (`raw_dir` or the import snapshot) |
| `source_sha256` / `sha256` | Hash of the raw source / of this file |
| `width`, `height` | Pixel size (after EXIF orientation) |
| `frame_index`, `timestamp_s` | For video frames |
| `parent_item_id` | For augmented images |

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundation, config, logging, manifests, ingestion (images + videos), CLI, tests | ✅ |
| 2 | Exact + perceptual-hash near-duplicate detection, duplicates report | ✅ |
| 3 | Pluggable annotation (YOLO backend), confidence/review policy, immutable run snapshots, review queue | ✅ |
| 4 | Source-aware deterministic train/val/test split with persistent history, conflict handling and deliberate resplit | ✅ |
| 4.5 | Legacy YOLO/Roboflow import, label provenance, review decisions, trusted-label resolution | ✅ |
| 5 | Albumentations augmentation of train only, bbox-safe, lineage, using resolved trusted labels | ✅ |
| 6 | YOLO dataset export + `dataset.yaml`, validation checks, JSON + Markdown report | ⏳ |
| 7 | Optional `train` command, end-to-end test, full documentation | ⏳ |

## Testing

```bash
pytest
```

Tests generate a tiny synthetic dataset (images in jpg/png/webp, videos in mp4/mov) in a
temp directory, so no real data is needed. Annotation tests use a deterministic scripted
backend, so no model is needed either.

There is also an optional Ultralytics integration smoke test, skipped by default. It
checks loading, schema, caching and CPU determinism with a stock COCO model. **It does not
measure annotation quality for any project.**

```bash
mkdir -p ~/.cache/cv-pipeline-smoke
curl -L -o ~/.cache/cv-pipeline-smoke/yolo11n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
CV_PIPELINE_SMOKE_MODEL=~/.cache/cv-pipeline-smoke/yolo11n.pt pytest tests/test_yolo_smoke.py
```
