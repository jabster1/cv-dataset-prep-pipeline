"""Pure-function tests: policy, identities, schema normalisation."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from cv_pipeline.annotation.identity import digest, input_pixel_sha256, make_run_id
from cv_pipeline.annotation.policy import (
    PolicySettings,
    audit_sampled,
    evaluate_boxes,
    item_review,
)
from cv_pipeline.annotation.schema import (
    BoxDecision,
    InferenceStatus,
    RawDetection,
    ReviewReason,
    ReviewStatus,
    canonical_detections,
)

from .fakes import det


def settings(**kw) -> PolicySettings:
    base = dict(
        classes=("lock", "tag", "hasp"),
        class_map=(),
        confidence_threshold=0.25,
        review_threshold=0.5,
        review_empty=True,
        review_unmapped_classes=False,
        review_sample_rate=0.0,
        seed=1,
    )
    base.update(kw)
    return PolicySettings(**base)


def decisions(dets, s=None, w=100, h=100):
    return evaluate_boxes(canonical_detections(dets), w, h, s or settings())


def test_threshold_boundaries() -> None:
    out = decisions(
        [
            det("lock", 0.2499, 0, 0, 10, 10),
            det("lock", 0.25, 0, 0, 10, 10),
            det("lock", 0.4999, 0, 0, 10, 10),
            det("lock", 0.5, 0, 0, 10, 10),
        ]
    )
    by_conf = {b.detection.confidence: b for b in out}
    assert by_conf[0.2499].decision == BoxDecision.DISCARDED_BELOW_MIN_CONFIDENCE
    assert by_conf[0.25].decision == BoxDecision.CANDIDATE and by_conf[0.25].review_flag
    assert by_conf[0.4999].review_flag
    assert by_conf[0.5].decision == BoxDecision.CANDIDATE and not by_conf[0.5].review_flag


def test_class_mapping_explicit_and_implicit() -> None:
    s = settings(class_map=(("padlock", "lock"),))
    assert s.class_id("padlock") == 0
    assert s.class_id("tag") == 1  # same name as a project class
    assert s.class_id("person") is None
    out = decisions([det("person", 0.9, 0, 0, 10, 10)], s)
    assert out[0].decision == BoxDecision.DISCARDED_UNMAPPED_CLASS


def test_invalid_and_clipped_boxes() -> None:
    out = decisions(
        [
            det("lock", 0.9, 50, 50, 40, 60),  # x2 < x1
            det("tag", 0.9, -10, -10, 50, 50),  # partly outside -> clipped
            det("hasp", 0.9, 200, 200, 300, 300),  # fully outside -> zero area
            det("lock", math.nan, 0, 0, 10, 10),
        ]
    )
    by_label = {}
    for b in out:
        by_label.setdefault(b.detection.label, []).append(b)
    assert all(b.decision == BoxDecision.DISCARDED_INVALID_BOX for b in by_label["lock"])
    tag = by_label["tag"][0]
    assert tag.decision == BoxDecision.CANDIDATE and tag.clipped and tag.xyxy == (0, 0, 50, 50)
    assert tag.yolo(100, 100) == (0.25, 0.25, 0.5, 0.5)
    assert by_label["hasp"][0].decision == BoxDecision.DISCARDED_INVALID_BOX


def test_item_review_reasons() -> None:
    s = settings()
    ok = InferenceStatus.OK
    assert item_review("a", ok, decisions([det("lock", 0.9, 0, 0, 9, 9)]), s) == (
        ReviewStatus.UNREVIEWED,
        [],
    )
    assert item_review("a", ok, decisions([det("lock", 0.3, 0, 0, 9, 9)]), s)[1] == [
        ReviewReason.LOW_CONFIDENCE
    ]
    assert item_review("a", ok, [], s)[1] == [ReviewReason.NO_DETECTIONS]
    assert item_review("a", ok, [], settings(review_empty=False))[0] == ReviewStatus.UNREVIEWED
    # only discarded boxes => no candidates => no_detections
    assert item_review("a", ok, decisions([det("lock", 0.1, 0, 0, 9, 9)]), s)[1] == [
        ReviewReason.NO_DETECTIONS
    ]
    unmapped = decisions([det("lock", 0.9, 0, 0, 9, 9), det("cat", 0.9, 0, 0, 9, 9)])
    assert item_review("a", ok, unmapped, s)[1] == []
    assert item_review("a", ok, unmapped, settings(review_unmapped_classes=True))[1] == [
        ReviewReason.UNMAPPED_CLASS
    ]
    assert item_review("a", InferenceStatus.FAILED, [], s) == (
        ReviewStatus.NEEDS_REVIEW,
        [ReviewReason.INFERENCE_FAILED],
    )
    assert item_review("a", InferenceStatus.UNREADABLE_IMAGE, [], s)[1] == [
        ReviewReason.UNREADABLE_IMAGE
    ]


def test_audit_sample_is_deterministic_and_roughly_rate() -> None:
    ids = [f"img_{i:05d}" for i in range(4000)]
    picked = [i for i in ids if audit_sampled(i, seed=7, rate=0.05)]
    assert picked == [i for i in ids if audit_sampled(i, seed=7, rate=0.05)]
    assert 0.035 < len(picked) / len(ids) < 0.065
    assert not any(audit_sampled(i, 7, 0.0) for i in ids)
    assert all(audit_sampled(i, 7, 1.0) for i in ids)
    assert picked != [i for i in ids if audit_sampled(i, seed=8, rate=0.05)]


def test_audit_sample_only_for_otherwise_unflagged_items() -> None:
    s = settings(review_sample_rate=1.0)
    status, reasons = item_review(
        "a", InferenceStatus.OK, decisions([det("lock", 0.9, 0, 0, 9, 9)]), s
    )
    assert (status, reasons) == (ReviewStatus.NEEDS_REVIEW, [ReviewReason.AUDIT_SAMPLE])
    assert item_review("a", InferenceStatus.OK, [], s)[1] == [ReviewReason.NO_DETECTIONS]


def test_canonical_detections_order_independent_and_rounded() -> None:
    a = [
        det("tag", 0.81234567, 1.23456, 2, 3, 4),
        det("lock", 0.5, 0, 0, 1, 1),
        det("lock", 0.9, 5, 5, 9, 9),
    ]
    assert canonical_detections(a) == canonical_detections(list(reversed(a)))
    first = canonical_detections(a)
    assert [d.label for d in first] == ["lock", "lock", "tag"]
    assert first[0].confidence == 0.9  # higher confidence first within a label
    assert first[2].confidence == 0.812346 and first[2].xyxy[0] == 1.235


def test_raw_detection_json_roundtrip_with_nan() -> None:
    d = RawDetection("x", 3, math.nan, (1.0, math.nan, 3.0, 4.0))
    back = RawDetection.from_json(d.to_json())
    assert back.label == "x" and back.label_id == 3
    assert math.isnan(back.confidence) and math.isnan(back.xyxy[1])


def test_input_pixel_sha256_matches_documented_spec() -> None:
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    expected = hashlib.sha256(
        b"cvp-input-pixels-v1\n" + b"2 3 3 uint8 BGR HWC C\n" + img.tobytes()
    ).hexdigest()
    assert input_pixel_sha256(img) == expected
    # same bytes, different shape -> different identity
    assert input_pixel_sha256(img.reshape(3, 2, 3)) != expected
    # non-contiguous views hash by content
    big = np.zeros((4, 6, 3), np.uint8)
    big[::2, ::2] = img
    assert input_pixel_sha256(big[::2, ::2]) == expected
    with pytest.raises(ValueError):
        input_pixel_sha256(np.zeros((2, 2), np.uint8))


def test_identity_helpers_are_stable() -> None:
    assert digest({"b": 1, "a": [1, 2]}) == digest({"a": [1, 2], "b": 1})
    rid = make_run_id("s", "n", "r")
    assert rid.startswith("ann_") and len(rid) == 20
    assert rid != make_run_id("s", "n", "r2")
    with pytest.raises(ValueError):
        digest({"x": math.nan})
