"""Ingestion stage: raw images/videos -> ``interim/images`` + ``manifests/items.csv``.

Guarantees
----------
* Raw files are only ever opened for reading.
* Every raw file ends up either in ``items.csv`` (possibly as many frames) or in
  ``rejected.csv`` with a reason — nothing is dropped silently.
* Among byte-identical raw files, the canonical copy is chosen by
  :func:`cv_pipeline.canonical.canonical_path_key` (e.g. ``images/x.jpg`` wins over
  ``images/duplicates/x.jpg``); the others are recorded as ``exact_duplicate``.
* IDs are content-addressed, so renaming/moving raw files does not change them, and
  re-running with the same inputs and config produces identical manifests.
* Re-runs are incremental: items whose outputs already exist are reused unless
  ``force=True``.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from cv_pipeline.canonical import canonical_path_key
from cv_pipeline.config import PipelineConfig
from cv_pipeline.ingest.images import ImageReadError, image_item_id, ingest_image
from cv_pipeline.ingest.video import VideoReadError, iter_frames, probe_video, select_frame_indices
from cv_pipeline.manifest import (
    RAW_ORIGIN,
    ItemRecord,
    RejectedRecord,
    RejectReason,
    SourceRefRecord,
    SourceType,
    import_origin,
    read_records,
    write_records,
)
from cv_pipeline.utils.hashing import sha256_bytes, sha256_file
from cv_pipeline.utils.io import atomic_write_bytes, atomic_write_json, encode_image, rel_posix
from cv_pipeline.utils.logging import get_logger, log_event
from cv_pipeline.workspace import Workspace

logger = get_logger("ingest")

_PROGRESS_EVERY = 250


@dataclass
class IngestResult:
    items: list[ItemRecord] = field(default_factory=list)
    rejected: list[RejectedRecord] = field(default_factory=list)
    images_ingested: int = 0
    videos_ingested: int = 0
    frames_extracted: int = 0
    reused: int = 0
    hidden_files_ignored: int = 0

    def summary(self) -> dict[str, object]:
        return {
            "items_total": len(self.items),
            "images_ingested": self.images_ingested,
            "videos_ingested": self.videos_ingested,
            "frames_extracted": self.frames_extracted,
            "rejected_total": len(self.rejected),
            "rejected_by_reason": dict(sorted(Counter(r.reason for r in self.rejected).items())),
            "hidden_files_ignored": self.hidden_files_ignored,
        }


def video_source_id(sha256: str) -> str:
    return f"vid_{sha256[:12]}"


def frame_item_id(video_sha256: str, frame_index: int) -> str:
    return f"{video_source_id(video_sha256)}_f{frame_index:06d}"


def discover_raw_files(raw_dir: Path) -> tuple[list[Path], int]:
    """All non-hidden files under ``raw_dir``, sorted by relative POSIX path."""
    files: list[Path] = []
    hidden = 0
    for p in raw_dir.rglob("*"):
        rel_parts = p.relative_to(raw_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            hidden += p.is_file()
            continue
        if p.is_file():
            files.append(p)
    files.sort(key=lambda p: p.relative_to(raw_dir).as_posix())
    return files, hidden


def run_ingest(cfg: PipelineConfig, force: bool = False) -> IngestResult:
    ws = Workspace(cfg.workspace_dir).ensure()
    raw_dir = cfg.raw_dir
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. Put images under {raw_dir}/images "
            f"and videos under {raw_dir}/videos (or set ingest.raw_dir)."
        )

    previous = _load_previous(ws) if not force else {}
    files, hidden = discover_raw_files(raw_dir)
    result = IngestResult(hidden_files_ignored=hidden)
    if hidden:
        log_event(logger, "hidden_files_ignored", count=hidden)

    image_exts = set(cfg.ingest.image_extensions)
    video_exts = set(cfg.ingest.video_extensions)
    t0 = time.perf_counter()
    log_event(logger, "ingest_started", raw_dir=str(raw_dir), files=len(files), force=force)

    # Pass 1: classify and hash every file from every origin (raw + configured imports),
    # so byte-identical files resolve to one canonical copy independent of discovery order.
    candidates: list[_Candidate] = []
    for src in files:
        rel = rel_posix(src, raw_dir)
        ext = src.suffix.lower()
        if ext not in image_exts and ext not in video_exts:
            result.rejected.append(
                RejectedRecord(
                    rel, RejectReason.UNSUPPORTED_EXTENSION, f"extension {ext or '<none>'}"
                )
            )
            log_event(
                logger, "file_rejected", logging.WARNING, file=rel, reason="unsupported_extension"
            )
            continue
        candidates.append(_Candidate(src, rel, sha256_file(src), RAW_ORIGIN, raw_dir, 0, ""))
    candidates.extend(_import_candidates(cfg, result))

    canonical: dict[str, _Candidate] = {}  # sha256 -> canonical copy
    for c in candidates:
        if c.sha not in canonical or c.key < canonical[c.sha].key:
            canonical[c.sha] = c

    # Pass 2: ingest canonical copies; record the others as exact duplicates.
    for n, c in enumerate(candidates, start=1):
        is_image = c.src.suffix.lower() in image_exts
        keep = canonical[c.sha]
        if keep is not c:
            keep_is_image = PurePosixPath(keep.rel).suffix.lower() in image_exts
            keep_id = image_item_id(c.sha) if keep_is_image else video_source_id(c.sha)
            where = keep.rel if keep.origin == RAW_ORIGIN else f"{keep.origin}/{keep.rel}"
            result.rejected.append(
                RejectedRecord(
                    c.rel,
                    RejectReason.EXACT_DUPLICATE,
                    f"byte-identical to {where}",
                    duplicate_of=keep_id,
                    origin=c.origin,
                )
            )
            log_event(
                logger,
                "exact_duplicate",
                logging.WARNING,
                file=c.rel,
                origin=c.origin,
                canonical=where,
            )
        elif is_image:
            _ingest_one_image(c.src, c.rel, c.sha, c.root, ws, previous, result, c.origin)
        else:
            _ingest_one_video(c.src, c.rel, c.sha, c.root, ws, cfg, previous, result)

        if n % _PROGRESS_EVERY == 0:
            log_event(logger, "ingest_progress", processed=n, total=len(candidates))

    write_records(
        ws.source_refs_manifest,
        _source_refs(candidates, canonical, result, image_exts),
        SourceRefRecord,
    )
    result.items.sort(key=lambda r: r.item_id)
    write_records(ws.items_manifest, result.items, ItemRecord)
    write_records(ws.rejected_manifest, result.rejected, RejectedRecord)
    atomic_write_json(ws.reports / "ingest_summary.json", result.summary())
    _warn_orphans(ws, result.items)

    log_event(
        logger,
        "ingest_finished",
        seconds=round(time.perf_counter() - t0, 2),
        reused=result.reused,
        **result.summary(),
    )
    return result


# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    src: Path
    rel: str  # relative to ``root``
    sha: str
    origin: str
    root: Path
    origin_rank: int  # raw first, then imports by name
    legacy_split: str

    @property
    def key(self) -> tuple[object, ...]:
        """Canonical-copy preference among byte-identical files (lower wins)."""
        return (self.origin_rank, *canonical_path_key(self.rel))


def _import_candidates(cfg: PipelineConfig, result: IngestResult) -> list[_Candidate]:
    """Validated images of every configured, imported dataset; bytes re-verified."""
    from cv_pipeline.imports.importer import load_imports  # local: avoids import cycle

    out = []
    for rank, ds in enumerate(load_imports(cfg), start=1):
        origin = import_origin(ds.name)
        for img in ds.images:
            src = ds.files_root / img.rel_path
            sha = sha256_file(src) if src.is_file() else ""
            if sha != img.sha256:
                result.rejected.append(
                    RejectedRecord(
                        img.rel_path,
                        RejectReason.IMPORT_FILE_MODIFIED,
                        "imported file missing or changed since import",
                        origin=origin,
                    )
                )
                log_event(
                    logger,
                    "file_rejected",
                    logging.WARNING,
                    file=img.rel_path,
                    origin=origin,
                    reason=RejectReason.IMPORT_FILE_MODIFIED,
                )
                continue
            out.append(
                _Candidate(src, img.rel_path, sha, origin, ds.files_root, rank, img.legacy_split)
            )
    return out


def _source_refs(
    candidates: list[_Candidate],
    canonical: dict[str, _Candidate],
    result: IngestResult,
    image_exts: set[str],
) -> list[SourceRefRecord]:
    present = {i.source_id for i in result.items}
    refs = []
    for c in candidates:
        is_image = c.src.suffix.lower() in image_exts
        source_id = image_item_id(c.sha) if is_image else video_source_id(c.sha)
        if source_id not in present:
            continue
        refs.append(
            SourceRefRecord(
                source_id=source_id,
                origin=c.origin,
                source_path=c.rel,
                source_sha256=c.sha,
                role="canonical" if canonical[c.sha] is c else "duplicate",
                legacy_split=c.legacy_split,
            )
        )
    refs.sort(key=lambda r: (r.source_id, r.role != "canonical", r.origin, r.source_path))
    return refs


def _load_previous(ws: Workspace) -> dict[str, ItemRecord]:
    if not ws.items_manifest.is_file():
        return {}
    try:
        return {r.item_id: r for r in read_records(ws.items_manifest, ItemRecord)}
    except (ValueError, KeyError) as exc:
        log_event(logger, "previous_manifest_unreadable", logging.WARNING, error=str(exc))
        return {}


def _reusable(rec: ItemRecord | None, ws: Workspace, source_sha: str) -> bool:
    return rec is not None and rec.source_sha256 == source_sha and (ws.root / rec.path).is_file()


def _ingest_one_image(
    src: Path,
    rel: str,
    sha: str,
    raw_dir: Path,
    ws: Workspace,
    previous: dict[str, ItemRecord],
    result: IngestResult,
    origin: str = RAW_ORIGIN,
) -> None:
    prev = previous.get(image_item_id(sha))
    if _reusable(prev, ws, sha):
        assert prev is not None
        prev.source_path = rel  # file may have been moved/renamed in raw/
        prev.origin = origin
        result.items.append(prev)
        result.images_ingested += 1
        result.reused += 1
        return
    try:
        rec = ingest_image(src, sha, raw_dir, ws.root, ws.interim_images)
    except ImageReadError as exc:
        result.rejected.append(
            RejectedRecord(rel, RejectReason.UNREADABLE_IMAGE, str(exc), origin=origin)
        )
        log_event(logger, "file_rejected", logging.WARNING, file=rel, reason="unreadable_image")
        return
    rec.origin = origin
    result.items.append(rec)
    result.images_ingested += 1
    log_event(logger, "image_ingested", logging.DEBUG, file=rel, item_id=rec.item_id)


def _ingest_one_video(
    src: Path,
    rel: str,
    sha: str,
    raw_dir: Path,
    ws: Workspace,
    cfg: PipelineConfig,
    previous: dict[str, ItemRecord],
    result: IngestResult,
) -> None:
    sampling = cfg.ingest.video
    try:
        info = probe_video(src)
    except VideoReadError as exc:
        result.rejected.append(RejectedRecord(rel, RejectReason.UNREADABLE_VIDEO, str(exc)))
        log_event(logger, "file_rejected", logging.WARNING, file=rel, reason="unreadable_video")
        return

    indices = select_frame_indices(info.frame_count, info.fps, sampling)
    expected = [previous.get(frame_item_id(sha, i)) for i in indices]
    if indices and all(_reusable(r, ws, sha) for r in expected):
        for rec in expected:
            assert rec is not None
            rec.source_path = rel
            result.items.append(rec)
        result.videos_ingested += 1
        result.frames_extracted += len(expected)
        result.reused += len(expected)
        log_event(logger, "video_reused", file=rel, frames=len(expected))
        return

    source_id = video_source_id(sha)
    ext = f".{sampling.image_format}"
    frames: list[ItemRecord] = []
    try:
        for idx, frame in iter_frames(src, indices):
            data = encode_image(frame, ext, sampling.jpeg_quality)
            item_id = frame_item_id(sha, idx)
            dest = ws.interim_images / f"{item_id}{ext}"
            atomic_write_bytes(dest, data)
            h, w = frame.shape[:2]
            frames.append(
                ItemRecord(
                    item_id=item_id,
                    path=rel_posix(dest, ws.root),
                    source_id=source_id,
                    source_type=SourceType.VIDEO_FRAME,
                    source_path=rel,
                    source_sha256=sha,
                    sha256=sha256_bytes(data),
                    width=int(w),
                    height=int(h),
                    frame_index=idx,
                    timestamp_s=round(idx / info.fps, 3),
                )
            )
    except VideoReadError as exc:
        result.rejected.append(RejectedRecord(rel, RejectReason.UNREADABLE_VIDEO, str(exc)))
        log_event(logger, "file_rejected", logging.WARNING, file=rel, reason="unreadable_video")
        return

    if not frames:
        result.rejected.append(
            RejectedRecord(
                rel,
                RejectReason.NO_FRAMES_EXTRACTED,
                f"{info.frame_count} frames reported, none decodable",
            )
        )
        log_event(logger, "file_rejected", logging.WARNING, file=rel, reason="no_frames_extracted")
        return

    result.items.extend(frames)
    result.videos_ingested += 1
    result.frames_extracted += len(frames)
    log_event(
        logger,
        "video_ingested",
        file=rel,
        source_id=source_id,
        native_fps=round(info.fps, 3),
        total_frames=info.frame_count,
        extracted=len(frames),
        mode=sampling.mode,
    )


def _warn_orphans(ws: Workspace, items: list[ItemRecord]) -> None:
    """Files left in interim/ from earlier runs (e.g. after changing sampling settings).

    They are not deleted automatically; they are ignored by later stages because only the
    manifest is authoritative.
    """
    referenced = {(ws.root / r.path).resolve() for r in items}
    orphans = [
        p for p in ws.interim_images.iterdir() if p.is_file() and p.resolve() not in referenced
    ]
    if orphans:
        log_event(
            logger,
            "orphaned_interim_files",
            logging.WARNING,
            count=len(orphans),
            dir=str(ws.interim_images),
            hint="not in items.csv; safe to delete, ignored by later stages",
        )
