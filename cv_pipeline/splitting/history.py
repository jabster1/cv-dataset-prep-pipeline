"""Persistent split-assignment history.

``splits/groups.csv``         one row per split group ever assigned (active or inactive)
``splits/group_sources.csv``  one row per source ever assigned: its group and split.
                              Source-level memory is what detects merges and restores
                              splits for data that disappears and later reappears.
``splits/revisions.jsonl``    append-only; one line per revision that changed anything.
                              Each line carries ``state_digest`` of the two CSVs above,
                              which is verified on load to detect interrupted writes.
``splits/items.csv``          materialised current view (rewritten every revision).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any

from cv_pipeline.annotation.identity import digest
from cv_pipeline.manifest import read_records, write_records
from cv_pipeline.utils.io import atomic_write_text
from cv_pipeline.workspace import Workspace

SPLITS = ("train", "val", "test")
PROTECTION = {"train": 1, "val": 2, "test": 3}  # higher = more protected hold-out


class Reason:
    ALLOCATED = "allocated"
    INHERITED = "inherited"
    RESPLIT = "resplit"
    CONFLICT_RESOLUTION = "conflict_resolution"


class SplitHistoryError(RuntimeError):
    """Persistent split history is missing pieces or inconsistent."""


@dataclass
class GroupHistoryRecord:
    split_group: str
    split: str
    status: str  # active | inactive
    n_sources: int
    source_set_sha256: str
    n_items: int
    item_set_sha256: str
    reason: str
    first_revision: int
    last_revision: int


@dataclass
class SourceHistoryRecord:
    source_id: str
    split_group: str
    split: str
    status: str  # active | inactive
    first_revision: int
    last_revision: int


@dataclass
class SplitItemRecord:
    item_id: str
    source_id: str
    source_type: str
    split_group: str
    split: str
    reason: str
    assigned_in_revision: int


@dataclass
class History:
    groups: dict[str, GroupHistoryRecord] = field(default_factory=dict)
    sources: dict[str, SourceHistoryRecord] = field(default_factory=dict)
    revisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def last_revision(self) -> dict[str, Any] | None:
        return self.revisions[-1] if self.revisions else None

    @property
    def next_revision(self) -> int:
        return len(self.revisions) + 1

    def state_digest(self) -> str:
        return state_digest(list(self.groups.values()), list(self.sources.values()))

    @property
    def empty(self) -> bool:
        return not self.revisions


def set_sha256(values: list[str] | tuple[str, ...]) -> str:
    return digest(sorted(values))


def state_digest(groups: list[GroupHistoryRecord], sources: list[SourceHistoryRecord]) -> str:
    return digest(
        {
            "groups": [asdict(g) for g in sorted(groups, key=lambda g: g.split_group)],
            "sources": [asdict(s) for s in sorted(sources, key=lambda s: s.source_id)],
        }
    )


def load_history(ws: Workspace) -> History:
    paths = (ws.split_groups_manifest, ws.split_sources_manifest, ws.split_revisions)
    present = [p.is_file() for p in paths]
    if not any(present):
        return History()
    if not all(present):
        missing = [p.name for p, ok in zip(paths, present, strict=True) if not ok]
        raise SplitHistoryError(
            f"Split history in {ws.splits} is incomplete (missing {missing}). Restore it from "
            "version control or an archive; it is never regenerated automatically."
        )
    history = History(
        groups={g.split_group: g for g in read_records(paths[0], GroupHistoryRecord)},
        sources={s.source_id: s for s in read_records(paths[1], SourceHistoryRecord)},
        revisions=[
            json.loads(line)
            for line in paths[2].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
    )
    last = history.last_revision
    if last is None or last.get("state_digest") != history.state_digest():
        raise SplitHistoryError(
            f"Split history in {ws.splits} does not match its last revision record "
            "(edited by hand or an interrupted write). Restore it before splitting."
        )
    return history


def save_history(
    ws: Workspace,
    groups: list[GroupHistoryRecord],
    sources: list[SourceHistoryRecord],
    items: list[SplitItemRecord],
    revisions: list[dict[str, Any]],
) -> None:
    """Write all split files. The revision log is written last so that an interrupted
    write is detected by ``load_history`` (digest mismatch)."""
    ws.splits.mkdir(parents=True, exist_ok=True)
    write_records(
        ws.split_groups_manifest, sorted(groups, key=lambda g: g.split_group), GroupHistoryRecord
    )
    write_records(
        ws.split_sources_manifest, sorted(sources, key=lambda s: s.source_id), SourceHistoryRecord
    )
    write_records(ws.split_items_manifest, sorted(items, key=lambda i: i.item_id), SplitItemRecord)
    atomic_write_text(
        ws.split_revisions,
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in revisions),
    )


def archive_history(ws: Workspace, revision: int) -> str:
    """Copy the current history into ``splits/archive/rev_<n>/`` (never deleted)."""
    dest = ws.splits / "archive" / f"rev_{revision:04d}"
    if dest.exists():
        raise SplitHistoryError(f"Archive {dest} already exists; refusing to overwrite it.")
    dest.mkdir(parents=True)
    for p in (
        ws.split_groups_manifest,
        ws.split_sources_manifest,
        ws.split_revisions,
        ws.split_items_manifest,
    ):
        if p.is_file():
            shutil.copy2(p, dest / p.name)
    return dest.relative_to(ws.root).as_posix()
