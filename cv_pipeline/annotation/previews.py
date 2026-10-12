"""Preview images: machine predictions drawn over the input image, for human reviewers.

* green  — candidate box
* amber  — candidate in the review range (low confidence)
* grey   — discarded box (below minimum confidence or unmapped), to help spot misses
"""

from __future__ import annotations

import cv2
import numpy as np

from cv_pipeline.annotation.policy import BoxOutcome
from cv_pipeline.annotation.schema import BoxDecision

_GREEN = (60, 180, 75)
_AMBER = (0, 165, 255)
_GREY = (160, 160, 160)


def render_preview(
    image: np.ndarray, boxes: list[BoxOutcome], classes: tuple[str, ...]
) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    thick = max(1, round(min(h, w) / 400))
    font_scale = max(0.4, min(h, w) / 1000)

    def draw(b: BoxOutcome, color: tuple[int, int, int], t: int) -> None:
        assert b.xyxy is not None
        x1, y1, x2, y2 = (int(round(v)) for v in b.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t, lineType=cv2.LINE_AA)
        name = classes[b.class_id] if b.class_id is not None else f"?{b.detection.label}"
        text = f"{name} {b.detection.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(y1, th + base)
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw, ty), color, -1)
        cv2.putText(
            out,
            text,
            (x1, ty - base),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    drawable = [b for b in boxes if b.xyxy is not None]
    for b in drawable:
        if b.decision != BoxDecision.CANDIDATE:
            draw(b, _GREY, 1)
    for b in drawable:
        if b.decision == BoxDecision.CANDIDATE:
            draw(b, _AMBER if b.review_flag else _GREEN, thick)
    return out
