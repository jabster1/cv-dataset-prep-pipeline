"""Create a small synthetic raw dataset so the pipeline can be tried without real data.

    python examples/make_synthetic_raw.py                 # -> data/example/raw
    python -m cv_pipeline prepare --config configs/example.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cv_pipeline.synthetic import write_synthetic_raw  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "example" / "raw")
    parser.add_argument("--images", type=int, default=24)
    parser.add_argument("--videos", type=int, default=3)
    parser.add_argument("--frames", type=int, default=60, help="frames per video")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stats = write_synthetic_raw(
        args.out,
        n_images=args.images,
        n_videos=args.videos,
        video_frames=args.frames,
        seed=args.seed,
        labels_dir=args.out.parent / "ground_truth_labels",
    )
    print(f"Wrote synthetic raw data to {args.out}: {stats}")


if __name__ == "__main__":
    main()
