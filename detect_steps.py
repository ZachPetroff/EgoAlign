#!/usr/bin/env python3

import csv
import json
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import shadow.fileio as sf

# ── Configuration ──────────────────────────────────────────────────────────────
BODY_NODES = [
    "Body", "Hips",
    "RightThigh", "RightLeg",  "RightFoot", "RightToe", "RightToeEnd", "RightHeel",
    "LeftThigh",  "LeftLeg",   "LeftFoot",  "LeftToe",  "LeftToeEnd",  "LeftHeel",
    "SpineLow", "SpineMid", "Chest",
    "RightShoulder", "RightArm", "RightForearm", "RightHand",
    "RightFinger", "RightFingerEnd",
    "LeftShoulder",  "LeftArm",  "LeftForearm",  "LeftHand",
    "LeftFinger",  "LeftFingerEnd",
    "Neck", "Head", "HeadEnd",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_position(frame, node_map, node_name):
    base = node_map[node_name]["c"][0]
    return (
        frame[base + 1] / 100.0,
        frame[base + 2] / 100.0,
        frame[base + 3] / 100.0,
    )

def get_pressure(frame, node_map, foot_name):
    sl = node_map[foot_name]["p"]
    return sum(max(0.0, frame[i]) for i in range(sl[0], sl[1]))

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Shadow MoCap data to CSV")
    parser.add_argument(
        "--shadow_dir",
        type=str,
        required=True,
        help="Directory containing data.mStream and take.mTake"
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV file path (e.g. detected_steps.csv)"
    )

    args = parser.parse_args()

    shadow_dir = Path(args.shadow_dir)
    out_csv = Path(args.out_csv)

    mstream_path = shadow_dir / "data.mStream"
    mtake_path   = shadow_dir / "take.mTake"

    if not mstream_path.exists() or not mtake_path.exists():
        raise FileNotFoundError(
            f"Missing required files in {shadow_dir}. "
            f"Expected data.mStream and take.mTake"
        )

    # ── Load data ──────────────────────────────────────────────────────────────
    with open(mstream_path, "rb") as f:
        info, node_list, data = sf.read_stream(f)

    with open(mtake_path) as f:
        take_meta = json.load(f)

    with open(mtake_path) as f:
        node_map = sf.make_node_map(f, node_list)

    num_frame  = info["num_frame"]
    frame_size = info["frame_stride"] // 4
    h          = info["h"]

    # Parse start time
    start_str = take_meta["start"].rstrip("Z")
    if "." in start_str:
        integer_part, frac = start_str.split(".")
        frac = frac[:6].ljust(6, "0")
        start_str = f"{integer_part}.{frac}"
    start_utc = datetime.fromisoformat(start_str)

    print(f"{num_frame:,} frames | h={h}s | start={start_utc}")

    # ── CSV fields ─────────────────────────────────────────────────────────────
    pose_fields = [f"{node}_{ax}" for node in BODY_NODES for ax in ("x", "y", "z")]
    fields = (
        ["frame", "time", "utc_time",
         "left_total_pressure", "right_total_pressure", "max_pressure_foot"]
        + pose_fields
    )

    # ── Process frames ─────────────────────────────────────────────────────────
    rows = []
    for fi in range(num_frame):
        frame = data[fi * frame_size : (fi + 1) * frame_size]

        left_p  = get_pressure(frame, node_map, "LeftFoot")
        right_p = get_pressure(frame, node_map, "RightFoot")
        max_foot = "LeftFoot" if left_p >= right_p else "RightFoot"

        t       = fi * h
        utc     = start_utc + timedelta(seconds=t)
        utc_str = utc.strftime("%Y-%m-%d %H:%M:%S.") + f"{utc.microsecond // 1000:03d}"

        row = {
            "frame": fi,
            "time": round(t, 10),
            "utc_time": utc_str,
            "left_total_pressure": left_p,
            "right_total_pressure": right_p,
            "max_pressure_foot": max_foot,
        }

        for node_name in BODY_NODES:
            x, y, z = get_position(frame, node_map, node_name)
            row[f"{node_name}_x"] = x
            row[f"{node_name}_y"] = y
            row[f"{node_name}_z"] = z

        rows.append(row)

    # ── Write CSV ──────────────────────────────────────────────────────────────
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done — {out_csv}")

if __name__ == "__main__":
    main()
