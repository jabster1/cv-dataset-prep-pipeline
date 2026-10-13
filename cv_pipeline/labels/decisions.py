"""Minimal human review decisions: ``review/decisions.csv`` (written by people or a future UI).

| column          | required | meaning                                                     |
|-----------------|----------|-------------------------------------------------------------|
| item_id         | yes      | reviewed item                                               |
| based_on_run_id | yes      | annotation run whose machine labels were reviewed           |
| decision        | yes      | accepted | corrected | rejected                             |
| reviewer        | no       | free text / id                                              |
| notes           | no       |                                                             |
| label_sha256    | no       | corrected only: sha256 of the edited review/labels file; if |
|                 |          | given it must match, otherwise the current file is recorded |

* ``accepted``  the machine label from that run is correct as-is (and the review label file
                must not have been edited — that would be ``corrected``)
* ``corrected`` ``review/labels/<item_id>.txt`` is the trusted annotation
* ``rejected``  the machine label from that run must not be used, under any policy

``based_on_run_id: manual`` (with ``corrected`` only) records a label drawn from scratch for
an item that never had a machine pre-annotation (e.g. annotation disabled or failed).

Rows are processed in file order; the last row for an item wins (reviewers may revise).
Review is never inferred from the existence of a label file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DECISIONS = ("accepted", "corrected", "rejected")
REQUIRED = ("item_id", "based_on_run_id", "decision")
COLUMNS = ("item_id", "based_on_run_id", "decision", "reviewer", "notes", "label_sha256")


@dataclass(frozen=True)
class ReviewDecision:
    row: int  # 1-based data row in the file, for audit messages
    item_id: str
    based_on_run_id: str
    decision: str
    reviewer: str = ""
    notes: str = ""
    label_sha256: str = ""


@dataclass(frozen=True)
class InvalidDecision:
    row: int
    item_id: str
    problem: str


def read_decisions(path: Path) -> tuple[dict[str, ReviewDecision], list[InvalidDecision]]:
    """Latest well-formed decision per item, plus rows that are malformed."""
    if not path.is_file():
        return {}, []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            return {}, [InvalidDecision(0, "", f"decisions.csv is missing columns {missing}")]
        latest: dict[str, ReviewDecision] = {}
        invalid: list[InvalidDecision] = []
        for i, row in enumerate(reader, start=1):
            item = (row.get("item_id") or "").strip()
            run = (row.get("based_on_run_id") or "").strip()
            decision = (row.get("decision") or "").strip().lower()
            if not item or not run:
                invalid.append(InvalidDecision(i, item, "item_id and based_on_run_id are required"))
                continue
            if decision not in DECISIONS:
                invalid.append(
                    InvalidDecision(i, item, f"decision {decision!r} not in {list(DECISIONS)}")
                )
                continue
            latest[item] = ReviewDecision(
                row=i,
                item_id=item,
                based_on_run_id=run,
                decision=decision,
                reviewer=(row.get("reviewer") or "").strip(),
                notes=(row.get("notes") or "").strip(),
                label_sha256=(row.get("label_sha256") or "").strip().lower(),
            )
    return latest, invalid
