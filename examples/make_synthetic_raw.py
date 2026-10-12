"""Create a small synthetic raw dataset so the pipeline can be tried without real data.

    python examples/make_synthetic_raw.py                 # -> data/example/raw
    python -m cv_pipeline prepare --config configs/example.yaml

By default a few known duplicates are planted so the duplicates report has content:
a renamed byte-identical copy, a pixel-identical re-save, and a brighter variant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import shutil  # noqa: E402

import cv2  # noqa: E402

from cv_pipeline.synthetic import write_synthetic_raw  # noqa: E402


def plant_duplicates(raw: Path) -> None:
    images = raw / "images"
    src = images / "scene_001.png"
    dup_dir = images / "duplicates"
    dup_dir.mkdir(exist_ok=True)
    shutil.copyfile(src, dup_dir / "renamed_copy_of_scene_001.png")  # byte-identical
    img = cv2.imread(str(src))
    cv2.imwrite(str(dup_dir / "resaved_scene_001.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    brighter = cv2.convertScaleAbs(cv2.imread(str(images / "scene_004.png")), alpha=1.0, beta=15)
    cv2.imwrite(str(dup_dir / "brighter_scene_004.jpg"), brighter, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "example" / "raw")
    parser.add_argument("--images", type=int, default=24)
    parser.add_argument("--videos", type=int, default=3)
    parser.add_argument("--frames", type=int, default=60, help="frames per video")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-duplicates", action="store_true", help="don't plant duplicates")
    args = parser.parse_args()

    stats = write_synthetic_raw(
        args.out,
        n_images=args.images,
        n_videos=args.videos,
        video_frames=args.frames,
        seed=args.seed,
        labels_dir=args.out.parent / "ground_truth_labels",
    )
    if not args.no_duplicates:
        plant_duplicates(args.out)
    print(f"Wrote synthetic raw data to {args.out}: {stats}")


if __name__ == "__main__":
    main()
