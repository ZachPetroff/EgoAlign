#!/usr/bin/env python3
"""
process_aria.py
===============
Extract RGB frames and per-frame eye-gaze projections from an Aria .vrs
recording using the MPS pipeline, and write them to disk.

Usage:
    python process_aria.py \
        --vrs      path/to/recording.vrs \
        --gaze-csv path/to/mps/eye_gaze/general_eye_gaze.csv \
        --out-dir  rgb_frames_with_gaze

Outputs (inside --out-dir):
    images/         – extracted Aria RGB frames (PNG, zero-padded filenames)
    frames.csv      – per-frame metadata:
                        frame_name, capture_timestamp_ns,
                        gaze_u_px, gaze_v_px, gaze_depth_m, has_gaze
"""

import argparse
import os
import csv

import cv2
import numpy as np
from tqdm import tqdm
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
)
from projectaria_tools.core.stream_id import StreamId
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Aria RGB frames and gaze projections to disk."
    )
    p.add_argument("--vrs", required=True,
                   help="Path to the Aria .vrs recording file.")
    p.add_argument("--gaze-csv", required=True,
                   help="Path to the MPS general_eye_gaze.csv file.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory (e.g. rgb_frames_with_gaze). "
                        "Will be created if it does not exist.")
    return p.parse_args()


def main():
    args = parse_args()

    vrs_path  = args.vrs
    gaze_path = args.gaze_csv
    out_dir   = args.out_dir
    img_dir   = os.path.join(out_dir, "images")
    csv_path  = os.path.join(out_dir, "frames.csv")
    os.makedirs(img_dir, exist_ok=True)

    provider = data_provider.create_vrs_data_provider(vrs_path)
    generalized_eye_gazes = mps.read_eyegaze(gaze_path)

    rgb_stream_id    = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    num_rgb_frames   = provider.get_num_data(rgb_stream_id)

    device_calibration = provider.get_device_calibration()
    camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    pad = max(5, len(str(max(0, num_rgb_frames - 1))))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_name",
            "capture_timestamp_ns",
            "gaze_u_px",
            "gaze_v_px",
            "gaze_depth_m",
            "has_gaze",
        ])

        for i in tqdm(range(num_rgb_frames)):
            rgb_frame = provider.get_image_data_by_index(rgb_stream_id, i)

            img_rgb = rgb_frame[0].to_numpy_array()
            ts_ns   = int(rgb_frame[1].capture_timestamp_ns)

            # Work in BGR for OpenCV drawing + saving
            if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_rgb.copy()

            gaze_u = gaze_v = depth_m = None
            has_gaze = 0

            generalized_eye_gaze = get_nearest_eye_gaze(generalized_eye_gazes, ts_ns)
            if generalized_eye_gaze is not None:
                depth_m = float(generalized_eye_gaze.depth) if generalized_eye_gaze.depth else 1.0
                uv = get_gaze_vector_reprojection(
                    generalized_eye_gaze,
                    rgb_stream_label,
                    device_calibration,
                    camera_calibration,
                    depth_m,
                )
                if uv is not None:
                    gaze_u, gaze_v = float(uv[0]), float(uv[1])
                    has_gaze = 1

                    frame_name = f"{i:0{pad}d}.png"
                    writer.writerow([frame_name, ts_ns, gaze_u, gaze_v, depth_m, has_gaze])

                    out_path = os.path.join(img_dir, frame_name)
                    ok = cv2.imwrite(out_path, img_bgr)
                    if not ok:
                        print(f"[WARN] Failed to write: {out_path}")

    print(f"Saved {num_rgb_frames} frames to: {img_dir}")
    print(f"Wrote CSV to: {csv_path}")


if __name__ == "__main__":
    main()
