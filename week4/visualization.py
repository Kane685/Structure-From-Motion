# import libraries
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np

# import visualization function
from sfm_utils import (
    visualize_open3d_scene
)

# set result path
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GF4 Week 4 visualization pipeline."
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        help="Path of result.",
    )
    args = parser.parse_args()
    return args

def run(args: argparse.Namespace) -> None:
    data = np.load(args.result_path, allow_pickle=True)

    kept_points   = data["points"]
    kept_colours  = data["colours"]
    cam_Rs        = data["cam_Rs"]
    cam_ts        = data["cam_ts"]
    cam_names     = data["cam_names"].tolist()

    camera_pose_list = [
        (name, R, t)
        for name, R, t in zip(cam_names, cam_Rs, cam_ts)
    ]

    visualize_open3d_scene(kept_points, kept_colours, camera_pose_list)

def main() -> int:
    args = parse_args()
    run(args)

if __name__ == "__main__":
    raise SystemExit(main())