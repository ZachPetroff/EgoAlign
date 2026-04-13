#!/usr/bin/env python3
"""
time_sync.py — Automatic DJI ↔ Shadow time synchronisation.

Usage:
    python time_sync.py \
        --srt_path  dji/DJI_20260228161138_0012_D.SRT \
        --steps_csv shadow/detected_steps.csv \
        --pose_dir  vitpose/vitpose_output \
        --out_csv   time_aligned_steps.csv

Algorithm (two-stage):
  1. COARSE  — convert the DJI local timestamp to UTC via the GPS-derived
               timezone.  This gives Δ = (DJI_true_UTC_t0 − Shadow_UTC_t0),
               which is used as the centre of the drift search window.
               Assumption: both devices were started at approximately the
               same time, so the unknown clock drift is close to Δ.

  2. FINE    — sweep drift candidates in [Δ − COARSE_MARGIN, Δ + COARSE_MARGIN]
               and maximise the Pearson correlation between:
                 • DJI:    heel_diff_y = rheel_y − lheel_y  (ViTPose pixel-Y)
                 • Shadow: zdiff       = RightHeel_z − LeftHeel_z  (world-Z)
               Both oscillate at the step frequency; the peak of |r| locates
               the exact drift.

Output:
  time_aligned_steps.csv — Shadow detected_steps rows, one per DJI frame,
  with two extra columns appended:
    matched_srt_time   UTC timestamp of the DJI frame (after drift correction)
    dji_frame          0-based DJI frame index
"""

import argparse
import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytz
from timezonefinder import TimezoneFinder

# ── Constants ─────────────────────────────────────────────────────────────────
COARSE_MARGIN = 20.0   # ± seconds around Δ for the coarse drift search
COARSE_STEP   =  0.033   # step size for the coarse search (seconds)
FINE_MARGIN   =  3.0   # ± seconds around the coarse peak for the fine search
FINE_STEP     =  0.001 # step size for the fine search (seconds)
CONF_THRESH   =  0.3   # ViTPose keypoint confidence threshold
L_HEEL_IDX    = 19     # COCO-WholeBody: left heel
R_HEEL_IDX    = 22     # COCO-WholeBody: right heel

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Load and timezone-convert DJI SRT
# ─────────────────────────────────────────────────────────────────────────────

def load_srt(srt_path: Path):
    """
    Parse the DJI .SRT sidecar file.

    Returns
    -------
    dji_local_times : list[datetime]  — per-frame local device timestamps
    dji_sec         : np.ndarray      — seconds since first frame
    lat, lon        : float           — GPS coordinates of the first frame
    """
    text = srt_path.read_text()

    ts_pat  = re.compile(r"FrameCnt:\s*\d+.*?\n([\d\-]+ [\d:\.]+)", re.DOTALL)
    gps_pat = re.compile(r"\[latitude:\s*([\d\.\-]+)\]\s*\[longitude:\s*([\d\.\-]+)\]")

    dji_local_times = [
        datetime.strptime(m.group(1).strip(), "%Y-%m-%d %H:%M:%S.%f")
        for m in ts_pat.finditer(text)
    ]
    gps_m = gps_pat.search(text)
    lat, lon = float(gps_m.group(1)), float(gps_m.group(2))

    t0 = dji_local_times[0]
    dji_sec = np.array([(t - t0).total_seconds() for t in dji_local_times])

    return dji_local_times, dji_sec, lat, lon


def local_to_utc(dji_local_times, lat, lon):
    """
    Convert DJI local timestamps to UTC using the GPS-derived timezone.

    Uses tz.localize() (correct historical DST handling) rather than
    datetime(..., tzinfo=tz) which gives the wrong LMT offset.

    Returns
    -------
    dji_t0_utc : datetime (UTC)
    utc        : pytz.UTC
    """
    tf      = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    tz      = pytz.timezone(tz_name)
    utc     = pytz.utc

    dji_t0_utc = tz.localize(dji_local_times[0]).astimezone(utc)
    utc_offset_s = tz.localize(dji_local_times[0]).utcoffset().total_seconds()

    print(f"  Timezone   : {tz_name}")
    print(f"  UTC offset : {utc_offset_s / 3600:+.1f} h")
    print(f"  DJI t0 UTC : {dji_t0_utc}")
    return dji_t0_utc, utc_offset_s, utc


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Load Shadow data and build signals
# ─────────────────────────────────────────────────────────────────────────────

def load_shadow(steps_csv: Path):
    """
    Load shadow/detected_steps.csv.

    Returns
    -------
    shadow_rows : list[dict]
    shadow_sec  : np.ndarray  — seconds since first row (UTC)
    shadow_t0   : datetime    — UTC timestamp of the first row
    shadow_zdiff: np.ndarray  — RightHeel_z − LeftHeel_z (world-Z)
    """
    rows = list(csv.DictReader(open(steps_csv)))

    def parse_utc(s):
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=pytz.utc)

    t0   = parse_utc(rows[0]["utc_time"])
    sec  = np.array([(parse_utc(r["utc_time"]) - t0).total_seconds() for r in rows])
    zdiff = np.array([float(r["RightHeel_z"]) - float(r["LeftHeel_z"]) for r in rows])

    print(f"  Shadow rows : {len(rows)}")
    print(f"  Shadow t0   : {t0}")
    print(f"  Shadow dur  : {sec[-1]:.3f} s")
    return rows, sec, t0, zdiff


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Load ViTPose heel keypoints
# ─────────────────────────────────────────────────────────────────────────────

def load_heel_diff(pose_dir: Path, n_frames: int):
    """
    Load per-frame heel pixel-Y positions from ViTPose JSON files
    (dji_*_keypoints.json).

    Returns
    -------
    heel_diff_y : np.ndarray  — rheel_y − lheel_y (NaN-interpolated)
    """
    lheel_y = np.full(n_frames, np.nan)
    rheel_y = np.full(n_frames, np.nan)

    for f in sorted(pose_dir.glob("dji_*_keypoints.json")):
        fi = int(f.stem.split("_")[1])
        if fi >= n_frames:
            continue
        with open(f) as fh:
            data = json.load(fh)
        people = data.get("people", [])
        if not people:
            continue
        kps = np.array(people[0]["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
        if kps[L_HEEL_IDX, 2] > CONF_THRESH:
            lheel_y[fi] = kps[L_HEEL_IDX, 1]
        if kps[R_HEEL_IDX, 2] > CONF_THRESH:
            rheel_y[fi] = kps[R_HEEL_IDX, 1]

    # Fill isolated NaN gaps by linear interpolation
    def fill_nans(arr):
        x, ok = np.arange(len(arr)), ~np.isnan(arr)
        return np.interp(x, x[ok], arr[ok])

    lheel_y = fill_nans(lheel_y)
    rheel_y = fill_nans(rheel_y)

    n_valid = np.sum(~np.isnan(lheel_y + rheel_y))
    print(f"  Pose files  : {n_valid} / {n_frames} valid frames")
    return rheel_y - lheel_y


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Drift estimation via Pearson correlation
# ─────────────────────────────────────────────────────────────────────────────

def pearson_at_drift(drift, dji_t, dji_sig, shadow_t, shadow_sig, Delta):
    """
    Pearson r between dji_sig and shadow_zdiff at a candidate drift value.

    For DJI frame at relative time t_d, the corresponding Shadow time is:
        shadow_t = t_d + (Delta − drift)
    """
    s_rel = dji_t + (Delta - drift)
    mask  = (s_rel >= shadow_t[0]) & (s_rel <= shadow_t[-1])
    if mask.sum() < 50:
        return 0.0
    d_win = dji_sig[mask]
    s_win = np.interp(s_rel[mask], shadow_t, shadow_sig)
    d_win = d_win - d_win.mean()
    s_win = s_win - s_win.mean()
    denom = np.sqrt(np.sum(d_win ** 2) * np.sum(s_win ** 2))
    return float(np.dot(d_win, s_win) / denom) if denom > 0 else 0.0


def estimate_drift(dji_sec, heel_diff_y, shadow_sec, shadow_zdiff, Delta):
    """
    Two-stage drift estimation.

    Stage 1: coarse scan centred at Δ (derived from recording start times).
    Stage 2: fine scan of ±FINE_MARGIN around the coarse peak.

    Returns
    -------
    best_drift : float  (seconds; DJI clock is this many seconds fast)
    best_r     : float  (peak Pearson r; should be ≈ −0.5)
    """
    # ── Coarse ────────────────────────────────────────────────────────────────
    # We assume data collection started at approximately the same time on both
    # devices.  The natural coarse estimate is drift ≈ Δ; we search ±COARSE_MARGIN.
    drift_coarse = np.arange(Delta - COARSE_MARGIN, Delta + COARSE_MARGIN, COARSE_STEP)
    r_coarse     = np.array([
        pearson_at_drift(d, dji_sec, heel_diff_y, shadow_sec, shadow_zdiff, Delta)
        for d in drift_coarse
    ])

    peak_coarse = drift_coarse[np.argmin(r_coarse)]   # correlation is negative
    print(f"  Coarse peak : {peak_coarse:.2f} s  (search: {Delta:.2f} ± {COARSE_MARGIN:.0f} s)")

    # ── Fine ──────────────────────────────────────────────────────────────────
    drift_fine = np.arange(peak_coarse - FINE_MARGIN, peak_coarse + FINE_MARGIN, FINE_STEP)
    r_fine     = np.array([
        pearson_at_drift(d, dji_sec, heel_diff_y, shadow_sec, shadow_zdiff, Delta)
        for d in drift_fine
    ])

    best_drift = drift_fine[np.argmin(r_fine)]
    best_r     = r_fine[np.argmin(r_fine)]

    print(f"  Fine peak   : {best_drift:.4f} s  (r = {best_r:.4f})")
    return best_drift, best_r


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Build and write time_aligned_steps.csv
# ─────────────────────────────────────────────────────────────────────────────

def build_aligned_csv(
    dji_local_times, dji_sec,
    shadow_rows, shadow_sec, shadow_t0,
    utc_offset_s, best_drift,
    out_csv: Path,
):
    """
    For each DJI frame, find the nearest Shadow row by UTC time and write the
    merged result.

    Columns = all Shadow detected_steps columns + matched_srt_time + dji_frame.
    """
    # UTC-corrected time for each DJI frame:
    #   DJI_true_UTC[i] = DJI_local[i] + UTC_offset − drift
    additive_offset = (-utc_offset_s) - best_drift   # = |UTC_offset| − drift
    dji_utc_offsets = np.array([
        (t - dji_local_times[0]).total_seconds() + additive_offset
        for t in dji_local_times
    ])  # seconds relative to DJI_local_t0

    dji_t0_utc_corrected = dji_local_times[0] + timedelta(seconds=additive_offset)
    dji_utc_datetimes    = [dji_t0_utc_corrected + timedelta(seconds=s)
                            for s in dji_sec]

    # Shadow UTC as seconds from shadow_t0 (already computed as shadow_sec)
    # For each DJI frame, find closest Shadow row by Shadow-relative UTC
    dji_shadow_times = np.array([
        (dt.replace(tzinfo=pytz.utc) - shadow_t0).total_seconds()
        for dt in dji_utc_datetimes
    ])

    fieldnames = list(shadow_rows[0].keys()) + ["matched_srt_time", "dji_frame"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        written = 0
        for fi, (t_shadow, dt_utc) in enumerate(zip(dji_shadow_times, dji_utc_datetimes)):
            if t_shadow < shadow_sec[0] or t_shadow > shadow_sec[-1]:
                continue   # DJI frame outside Shadow recording window
            nearest_idx = int(np.argmin(np.abs(shadow_sec - t_shadow)))
            row = dict(shadow_rows[nearest_idx])
            row["matched_srt_time"] = dt_utc.replace(tzinfo=pytz.utc).strftime(
                "%Y-%m-%d %H:%M:%S.%f+00:00"
            )
            row["dji_frame"] = fi
            writer.writerow(row)
            written += 1

    print(f"  Wrote {written} rows → {out_csv}")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute DJI↔Shadow time offset and produce time_aligned_steps.csv"
    )
    p.add_argument("--srt_path",  type=Path, required=True,
                   help="DJI .SRT sidecar file")
    p.add_argument("--steps_csv", type=Path, required=True,
                   help="Shadow detected_steps.csv")
    p.add_argument("--pose_dir",  type=Path, required=True,
                   help="Directory containing dji_*_keypoints.json files")
    p.add_argument("--out_csv",   type=Path, required=True,
                   help="Output CSV path (e.g. time_aligned_steps.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    print("── Stage 1: Parse DJI SRT and convert to UTC ──────────────────────")
    dji_local_times, dji_sec, lat, lon = load_srt(args.srt_path)
    dji_t0_utc, utc_offset_s, utc      = local_to_utc(dji_local_times, lat, lon)
    print(f"  DJI frames  : {len(dji_local_times)}")

    print("\n── Stage 2: Load Shadow data ───────────────────────────────────────")
    shadow_rows, shadow_sec, shadow_t0, shadow_zdiff = load_shadow(args.steps_csv)

    # Δ: how many seconds after Shadow start the DJI recording began (in UTC)
    Delta = (dji_t0_utc - shadow_t0).total_seconds()
    print(f"\n  Δ = DJI_true_UTC_t0 − Shadow_t0 = {Delta:.3f} s")
    print(  "  (Coarse drift estimate; fine search will refine this)")

    print("\n── Stage 3: Load ViTPose heel keypoints ────────────────────────────")
    heel_diff_y = load_heel_diff(args.pose_dir, len(dji_local_times))

    print("\n── Stage 4: Estimate clock drift via heel-height correlation ───────")
    best_drift, best_r = estimate_drift(
        dji_sec, heel_diff_y, shadow_sec, shadow_zdiff, Delta
    )
    additive_offset = (-utc_offset_s) - best_drift
    print(f"\n  Clock drift    : {best_drift:.4f} s (DJI clock fast by this amount)")
    print(f"  Additive offset: +{additive_offset:.3f} s  "
          f"→  DJI_true_UTC = DJI_local + {additive_offset:.3f} s")

    print("\n── Stage 5: Write time_aligned_steps.csv ───────────────────────────")
    build_aligned_csv(
        dji_local_times, dji_sec,
        shadow_rows, shadow_sec, shadow_t0,
        utc_offset_s, best_drift,
        args.out_csv,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
