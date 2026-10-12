"""Raw-inference cache keyed by (inference fingerprint, input pixel identity).

Independent of annotation run identity, so unchanged pixels are never re-inferred across
new dataset snapshots or policy changes. Only successful inferences are cached.
"""

from __future__ import annotations

import json
from pathlib import Path

from cv_pipeline.annotation.schema import SCHEMA_VERSION, RawDetection
from cv_pipeline.utils.io import atomic_write_json


class InferenceCache:
    def __init__(self, root: Path, inference_fp: str) -> None:
        self.inference_fp = inference_fp
        self.dir = root / inference_fp[:16]

    def _path(self, pixel_sha: str) -> Path:
        return self.dir / pixel_sha[:2] / f"{pixel_sha}.json"

    def get(self, pixel_sha: str) -> list[RawDetection] | None:
        path = self._path(pixel_sha)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            data.get("schema") != SCHEMA_VERSION
            or data.get("inference_fp") != self.inference_fp
            or data.get("input_pixel_sha256") != pixel_sha
        ):
            return None
        return [RawDetection.from_json(d) for d in data["detections"]]

    def put(self, pixel_sha: str, detections: list[RawDetection]) -> None:
        atomic_write_json(
            self._path(pixel_sha),
            {
                "schema": SCHEMA_VERSION,
                "inference_fp": self.inference_fp,
                "input_pixel_sha256": pixel_sha,
                "detections": [d.to_json() for d in detections],
            },
        )
