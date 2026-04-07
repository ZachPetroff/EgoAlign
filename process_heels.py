#!/usr/bin/env python3
"""
process_heels.py
================
Full pipeline for computing per-step median heel ground-contact positions.

Usage:
  python process_heels.py [options]

  --pose_dir  DIR   Directory of dji_*_keypoints.json ViTPose files
                    (default: ../vitpose/vitpose_output)
  --steps_csv FILE  time_aligned_steps.csv path
                    (default: ../time_aligned_steps.csv)
  --ply_path  FILE  3D mesh PLY path  (default: ../dji/recon_1.ply)
  --json_path FILE  NeRF-format camera JSON path  (default: ../dji/recon_1.json)
  --out_csv   FILE  Output CSV path  (default: ../median_heel_positions.csv)

Input files (defaults are all relative to the parent of this script's directory):
  vitpose/vitpose_output/dji_*_keypoints.json  – ViTPose 2D keypoints
  time_aligned_steps.csv                        – aligned Shadow MoCap data
  dji/recon_1.ply                               – 3D reconstruction mesh
  dji/recon_1.json                              – NeRF-format camera parameters

Output:
  ../median_heel_positions.csv                  – one row per footstep

Keypoint indices (COCO-WholeBody 133-keypoint format):
  Left heel  = kp[19]
  Right heel = kp[22]

Camera model (recon_1.json, NeRF/OpenGL convention):
  fl_x, fl_y, cx, cy  – per-frame intrinsics in pixels
  transform_matrix    – 4×4 camera-to-world (c2w), Y-up
  COORD_FIX = Rx(-90°) converts NeRF Y-up → mesh Z-up
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

DJI_FPS     = 30.0
CONF_THRESH = 0.0      # set >0 to skip low-confidence keypoints
L_HEEL_IDX  = 19      # COCO-WholeBody left heel
R_HEEL_IDX  = 22      # COCO-WholeBody right heel

SEARCH_RADIUS        = 0.5   # XY KDTree query radius at each ray sample (m)
N_RAY_SAMPLES        = 20    # points to sample along the ray per frame
Z_PADDING            = 0.5   # extra metres added above/below mesh Z bounds
GROUND_NORMAL_THRESH = 0.7   # |face_normal_z| > this → "ground" face
RAY_EPSILON          = 1e-9  # Möller–Trumbore near-zero threshold

# Coordinate fix: NeRF/OpenGL Y-up → Z-up (mesh frame). Rx(-90°).
COORD_FIX = np.array([[1,  0,  0],
                       [0,  0,  1],
                       [0, -1,  0]], dtype=np.float64)

WINDOW     = 10    # sliding window half-width (total = 2*WINDOW + 1 = 21)
MAD_FACTOR = 10.0  # flag if deviation > MAD_FACTOR × MAD

# Joint order for the (n_frames, n_joints, 3) skeleton array.
# Must contain RightHeel and LeftHeel.
JOINT_NAMES = [
    "Body", "Hips",
    "SpineLow", "SpineMid", "Chest", "Neck", "Head", "HeadEnd",
    "RightThigh", "RightLeg", "RightFoot", "RightToe", "RightToeEnd", "RightHeel",
    "LeftThigh",  "LeftLeg",  "LeftFoot",  "LeftToe",  "LeftToeEnd",  "LeftHeel",
    "RightShoulder", "RightArm", "RightForearm", "RightHand",
    "RightFinger", "RightFingerEnd",
    "LeftShoulder",  "LeftArm",  "LeftForearm",  "LeftHand",
    "LeftFinger", "LeftFingerEnd",
]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 – Extract heel 2D pixel positions + Shadow MoCap 3D positions
# ══════════════════════════════════════════════════════════════════════════════

def _load_steps(path):
    """Load time_aligned_steps.csv as a dict keyed by dji_frame (int)."""
    frame_map = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            frame_map[int(r["dji_frame"])] = r
    return frame_map


def _load_pose(frame_idx):
    """
    Load ViTPose keypoints for one DJI frame.
    Returns kps_array shaped (133, 3), or None if file missing / no people.
    kps_array[i] = [x, y, confidence] in 4K image coordinates.
    """
    path = POSE_DIR / f"dji_{frame_idx:012d}_keypoints.json"
    if not path.exists():
        return None
    with open(path) as fh:
        d = json.load(fh)
    people = d.get("people", [])
    if not people:
        return None
    raw = people[0]["pose_keypoints_2d"]
    return np.array(raw, dtype=np.float32).reshape(-1, 3)


def extract_heel_data():
    """
    Stage 1: build heel rows from ViTPose JSON files + time_aligned_steps.csv.

    Returns
    -------
    list[dict] with keys:
        dji_frame, video_t, max_pressure_foot,
        heel_x, heel_y, heel_conf,
        shadow_x, shadow_y, shadow_z
    """
    print(f"[Stage 1] Loading {STEPS_CSV.name} …")
    frame_map = _load_steps(STEPS_CSV)
    print(f"  {len(frame_map):,} aligned frames loaded")

    n_pose_files = sum(1 for f in POSE_DIR.iterdir() if f.name.startswith("dji_"))
    print(f"  Found {n_pose_files:,} ViTPose files in {POSE_DIR.name}/")

    results    = []
    n_no_pose  = 0
    n_no_row   = 0
    n_low_conf = 0

    for frame_idx in range(n_pose_files):
        s = frame_map.get(frame_idx)
        if s is None:
            n_no_row += 1
            continue

        stance = s["max_pressure_foot"]   # "LeftFoot" or "RightFoot"

        kps = _load_pose(frame_idx)
        if kps is None:
            n_no_pose += 1
            continue

        heel_idx = L_HEEL_IDX if stance == "LeftFoot" else R_HEEL_IDX
        hx, hy, hc = kps[heel_idx]

        if hc < CONF_THRESH:
            n_low_conf += 1
            continue

        side = "Left" if stance == "LeftFoot" else "Right"
        sx = float(s[f"{side}Heel_x"])
        sy = float(s[f"{side}Heel_y"])
        sz = float(s[f"{side}Heel_z"])

        results.append({
            "dji_frame":         frame_idx,
            "video_t":           f"{frame_idx / DJI_FPS:.4f}",
            "max_pressure_foot": stance,
            "heel_x":            f"{hx:.2f}",
            "heel_y":            f"{hy:.2f}",
            "heel_conf":         f"{hc:.4f}",
            "shadow_x":          f"{sx:.6f}",
            "shadow_y":          f"{sy:.6f}",
            "shadow_z":          f"{sz:.6f}",
        })

    print(f"  {len(results):,} heel rows extracted  "
          f"(no_row={n_no_row}, no_pose={n_no_pose}, low_conf={n_low_conf})")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 – Ray-mesh intersection
# ══════════════════════════════════════════════════════════════════════════════

def _load_ply(path):
    """Return (verts float32 N×3, faces int32 M×3)."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_verts = n_faces = 0
        vert_props = []
        in_vert = False
        for l in header_lines:
            toks = l.split()
            if l.startswith("element vertex"):
                n_verts = int(toks[-1]); in_vert = True
            elif l.startswith("element") and not l.startswith("element vertex"):
                if l.startswith("element face"):
                    n_faces = int(toks[-1])
                in_vert = False
            elif l.startswith("property") and in_vert:
                vert_props.append(toks[-1])

        bpv   = len(vert_props) * 4          # all float32
        raw_v = f.read(n_verts * bpv)
        arr   = np.frombuffer(raw_v, dtype="<f4").reshape(n_verts, len(vert_props))
        verts = arr[:, :3].copy()            # float32 xyz

        # Faces: 1 byte (count=3) + 3 × int32 = 13 bytes each
        raw_f = f.read(n_faces * 13)
        farr  = np.frombuffer(raw_f, dtype=np.uint8).reshape(n_faces, 13)
        faces = farr[:, 1:].view("<i4").reshape(n_faces, 3).copy().astype(np.int32)

    return verts, faces


def _load_cameras(path):
    """Return dict: frame_index → {pos, R_raw, fl_x, fl_y, cx, cy}."""
    with open(path) as f:
        data = json.load(f)

    cam_by_frame = {}
    for entry in data["frames"]:
        M     = np.array(entry["transform_matrix"], dtype=np.float64)
        pos   = COORD_FIX @ M[:3, 3]
        R_raw = M[:3, :3]

        fp = entry["file_path"].replace("\\", "/").split("/")[-1]
        try:
            frame_idx = int(fp.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue

        cam_by_frame[frame_idx] = {
            "pos":   pos,
            "R_raw": R_raw,
            "fl_x":  float(entry["fl_x"]),
            "fl_y":  float(entry["fl_y"]),
            "cx":    float(entry["cx"]),
            "cy":    float(entry["cy"]),
        }
    return cam_by_frame


def _ray_triangles_intersect(origin, direction, v0, v1, v2):
    """
    Vectorised Möller–Trumbore: test one ray against N triangles.

    Parameters
    ----------
    origin    : (3,) float64
    direction : (3,) float64  (need not be normalised)
    v0, v1, v2: (N, 3) float32

    Returns
    -------
    t    : (N,) float64  parametric distance (inf where no hit)
    valid: (N,) bool     True where a valid forward intersection exists
    """
    e1 = (v1 - v0).astype(np.float64)
    e2 = (v2 - v0).astype(np.float64)

    h     = np.cross(direction, e2)
    a     = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(a) > RAY_EPSILON
    inv_a = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)

    s = origin - v0.astype(np.float64)
    u = inv_a * np.einsum("ij,ij->i", s, h)
    valid &= (u >= 0.0) & (u <= 1.0)

    q = np.cross(s, e1)
    v = inv_a * np.einsum("j,ij->i", direction, q)
    valid &= (v >= 0.0) & (u + v <= 1.0)

    t = inv_a * np.einsum("ij,ij->i", e2, q)
    valid &= (t > RAY_EPSILON)

    return np.where(valid, t, np.inf), valid


def _no_hit_row(row):
    return {
        "dji_frame":         row["dji_frame"],
        "video_t":           row["video_t"],
        "max_pressure_foot": row["max_pressure_foot"],
        "heel_x":            row["heel_x"],
        "heel_y":            row["heel_y"],
        "shadow_x":          row["shadow_x"],
        "shadow_y":          row["shadow_y"],
        "shadow_z":          row["shadow_z"],
        "hit_x":             "nan",
        "hit_y":             "nan",
        "hit_z":             "nan",
        "hit_is_ground":     False,
        "n_candidates":      0,
    }


def intersect_heels(heel_rows):
    """
    Stage 2: ray-cast each heel pixel against the 3D mesh.

    Parameters
    ----------
    heel_rows : list[dict]  output of extract_heel_data()

    Returns
    -------
    list[dict] with keys:
        dji_frame, video_t, max_pressure_foot,
        heel_x, heel_y,
        shadow_x, shadow_y, shadow_z,
        hit_x, hit_y, hit_z, hit_is_ground, n_candidates
    """
    print(f"\n[Stage 2] Loading mesh from {PLY_PATH.name} …")
    verts, faces = _load_ply(PLY_PATH)
    print(f"  {len(verts):,} vertices,  {len(faces):,} faces")

    mesh_z_min = float(verts[:, 2].min())
    mesh_z_max = float(verts[:, 2].max())
    z_lo = mesh_z_min - Z_PADDING
    z_hi = mesh_z_max + Z_PADDING
    print(f"  Mesh Z range: {mesh_z_min:.2f} – {mesh_z_max:.2f} m  "
          f"(sweep: {z_lo:.2f} – {z_hi:.2f} m)")

    print("  Computing face normals …")
    fv0 = verts[faces[:, 0]].astype(np.float64)
    fv1 = verts[faces[:, 1]].astype(np.float64)
    fv2 = verts[faces[:, 2]].astype(np.float64)
    fn  = np.cross(fv1 - fv0, fv2 - fv0).astype(np.float32)
    fl  = np.linalg.norm(fn, axis=1, keepdims=True)
    ok  = fl[:, 0] > 1e-9
    fn[ok] /= fl[ok].astype(np.float32)

    print("  Building 2D KDTree on face centres …")
    centres_xy = ((verts[faces[:, 0], :2] +
                   verts[faces[:, 1], :2] +
                   verts[faces[:, 2], :2]) / 3.0).astype(np.float32)
    kd = cKDTree(centres_xy)
    print("  Done.")

    cam_by_frame = _load_cameras(JSON_PATH)
    print(f"  {len(cam_by_frame):,} cameras loaded")
    print(f"  {len(heel_rows):,} heel rows to intersect")

    print("\n  Intersecting rays …")
    results = []
    n_hit = n_ground_hit = n_miss = n_no_cam = total_cands = 0

    for i, row in enumerate(heel_rows):
        frame_idx = int(row["dji_frame"])
        cam = cam_by_frame.get(frame_idx)
        if cam is None:
            n_no_cam += 1
            continue

        # Build ray (NeRF/OpenGL convention → Z-up mesh frame)
        heel_x  = float(row["heel_x"])
        heel_y  = float(row["heel_y"])
        x_cam   = (heel_x - cam["cx"]) / cam["fl_x"]
        y_cam   = -(heel_y - cam["cy"]) / cam["fl_y"]   # flip Y: image↓ → cam↑
        d_cam   = np.array([x_cam, y_cam, -1.0])
        d_world = COORD_FIX @ (cam["R_raw"] @ d_cam)
        d_world /= np.linalg.norm(d_world)
        origin  = cam["pos"]

        # Sweep the ray across the full mesh Z range
        if abs(d_world[2]) > 1e-6:
            t_lo    = (z_lo - origin[2]) / d_world[2]
            t_hi    = (z_hi - origin[2]) / d_world[2]
            t_enter = max(0.0, min(t_lo, t_hi))
            t_exit  = max(t_lo, t_hi)
        else:
            t_enter, t_exit = 0.0, 30.0   # near-horizontal ray

        sample_ts = np.linspace(t_enter, t_exit, N_RAY_SAMPLES)
        all_cands: set = set()
        for t_s in sample_ts:
            xy_s = (origin + t_s * d_world)[:2]
            all_cands.update(kd.query_ball_point(xy_s, r=SEARCH_RADIUS))

        cand_idx = np.array(list(all_cands), dtype=np.int32)

        if len(cand_idx) == 0:
            n_miss += 1
            results.append(_no_hit_row(row))
            continue

        total_cands += len(cand_idx)
        cv0 = verts[faces[cand_idx, 0]]
        cv1 = verts[faces[cand_idx, 1]]
        cv2 = verts[faces[cand_idx, 2]]
        t_vals, valid = _ray_triangles_intersect(origin, d_world, cv0, cv1, cv2)

        if not valid.any():
            n_miss += 1
            results.append(_no_hit_row(row))
            continue

        n_hit += 1

        # Prefer ground-normal hits (|nz| > GROUND_NORMAL_THRESH)
        hit_normals  = fn[cand_idx]
        is_ground    = np.abs(hit_normals[:, 2]) > GROUND_NORMAL_THRESH
        ground_valid = valid & is_ground

        if ground_valid.any():
            best_t        = np.min(t_vals[ground_valid])
            hit_is_ground = True
            n_ground_hit += 1
        else:
            best_t        = np.min(t_vals[valid])
            hit_is_ground = False

        hit_pt = origin + best_t * d_world
        results.append({
            "dji_frame":         row["dji_frame"],
            "video_t":           row["video_t"],
            "max_pressure_foot": row["max_pressure_foot"],
            "heel_x":            row["heel_x"],
            "heel_y":            row["heel_y"],
            "shadow_x":          row["shadow_x"],
            "shadow_y":          row["shadow_y"],
            "shadow_z":          row["shadow_z"],
            "hit_x":             f"{hit_pt[0]:.6f}",
            "hit_y":             f"{hit_pt[1]:.6f}",
            "hit_z":             f"{hit_pt[2]:.6f}",
            "hit_is_ground":     hit_is_ground,
            "n_candidates":      len(cand_idx),
        })

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(heel_rows)}  hits={n_hit}  misses={n_miss}  "
                  f"avg_cands={total_cands//(i+1)}")

    print(f"  Ground hits={n_ground_hit:,}  other={n_hit - n_ground_hit:,}  "
          f"misses={n_miss:,}  no_cam={n_no_cam:,}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 – Sliding-window MAD outlier filtering
# ══════════════════════════════════════════════════════════════════════════════

def _sliding_median(arr, half_w):
    """Apply a 1D sliding median with window 2*half_w+1, mode='nearest'."""
    return median_filter(arr.astype(np.float64), size=2 * half_w + 1,
                         mode="nearest")


def _filter_foot(rows):
    """
    Compute a keep-mask for one foot's valid hit rows using sliding MAD.

    Parameters
    ----------
    rows : list[dict]  valid (non-NaN) hit rows for one foot, sorted by frame

    Returns
    -------
    keep : (N,) bool array  True = keep, False = outlier
    """
    xyz = np.array([[float(r["hit_x"]), float(r["hit_y"]), float(r["hit_z"])]
                    for r in rows])

    med_x = _sliding_median(xyz[:, 0], WINDOW)
    med_y = _sliding_median(xyz[:, 1], WINDOW)
    med_z = _sliding_median(xyz[:, 2], WINDOW)

    deviations = np.sqrt((xyz[:, 0] - med_x) ** 2 +
                         (xyz[:, 1] - med_y) ** 2 +
                         (xyz[:, 2] - med_z) ** 2)

    mad = np.median(np.abs(deviations - np.median(deviations)))
    threshold = MAD_FACTOR * max(mad, 1e-3)
    keep = deviations <= threshold

    print(f"    median_dev={np.median(deviations):.4f} m  "
          f"MAD={mad:.4f} m  threshold={threshold:.4f} m  "
          f"kept={keep.sum()}/{len(keep)}  outliers={(~keep).sum()}")
    return keep


def filter_intersections(all_rows):
    """
    Stage 3: apply sliding-window MAD filter to remove spatial outliers.

    Parameters
    ----------
    all_rows : list[dict]  output of intersect_heels()

    Returns
    -------
    list[dict]  same schema as input plus an 'outlier' bool field;
                outlier rows have hit_x/y/z replaced with 'nan'.
    """
    print(f"\n[Stage 3] Filtering {len(all_rows):,} intersection rows …")
    outlier_frames: set = set()

    for foot in ("LeftFoot", "RightFoot"):
        hit_rows = sorted(
            [r for r in all_rows
             if r["max_pressure_foot"] == foot and r["hit_x"] != "nan"],
            key=lambda r: int(r["dji_frame"])
        )
        if not hit_rows:
            continue
        print(f"  {foot}: {len(hit_rows)} valid hits")
        keep = _filter_foot(hit_rows)
        for i, r in enumerate(hit_rows):
            if not keep[i]:
                outlier_frames.add((foot, int(r["dji_frame"])))

    output_rows = []
    n_nullified = 0
    for r in all_rows:
        out = dict(r)
        key = (r["max_pressure_foot"], int(r["dji_frame"]))
        if key in outlier_frames:
            out["hit_x"]         = "nan"
            out["hit_y"]         = "nan"
            out["hit_z"]         = "nan"
            out["hit_is_ground"] = False
            out["outlier"]       = True
            n_nullified += 1
        else:
            out["outlier"] = False
        output_rows.append(out)

    n_hits_after = sum(1 for r in output_rows
                       if r["hit_x"] != "nan" and not r["outlier"])
    print(f"  Outliers nullified={n_nullified:,}  "
          f"valid hits remaining={n_hits_after:,}")
    return output_rows


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 – Per-step median heel positions + skateboarding correction
# ══════════════════════════════════════════════════════════════════════════════

def _detect_steps(df):
    """
    Return an (N, 3) int array of heel strikes.
    Each row: [row_index_in_df, dji_frame, stance_leg]
    stance_leg: 1 = RightFoot, 0 = LeftFoot.
    """
    foot_col    = df["max_pressure_foot"].values
    change_mask = np.concatenate([[True], foot_col[1:] != foot_col[:-1]])
    change_frames = np.where(change_mask)[0]

    steps = []
    for fi in change_frames:
        dji_frame  = int(df["dji_frame"].iloc[fi])
        stance_leg = 1 if foot_col[fi] == "RightFoot" else 0
        steps.append([fi, dji_frame, stance_leg])
    return np.array(steps, dtype=int)


def _fix_skateboarding(points_trimmed, parts, all_steps):
    """
    Correct foot-slippage artifacts in the Shadow MoCap data.

    For each step window, the stance heel is locked to its position at heel
    strike; the resulting rigid offset is applied to all joints in the window
    and carried forward.

    Parameters
    ----------
    points_trimmed : (n_frames, n_joints, 3) ndarray
    parts          : list[str]  joint names (axis 1)
    all_steps      : (N, 3) int ndarray  [row_idx, dji_frame, stance_leg]

    Returns
    -------
    shadow_fixed : (n_frames, n_joints, 3) ndarray
    """
    shadow_fixed = points_trimmed.copy()
    rheel_idx = parts.index("RightHeel")
    lheel_idx = parts.index("LeftHeel")
    hs         = all_steps[:, 0].astype(int)
    stance_leg = all_steps[:, 2].astype(int)

    for ss in range(len(hs) - 1):
        start = hs[ss]
        end   = hs[ss + 1]
        fr    = np.arange(start, end)

        heel_idx     = rheel_idx if stance_leg[ss] == 1 else lheel_idx
        foothold_xyz = shadow_fixed[start, heel_idx, :]        # (3,)
        curr_heel    = shadow_fixed[fr,    heel_idx, :]        # (len(fr), 3)
        slippage     = foothold_xyz[None, :] - curr_heel       # (len(fr), 3)

        shadow_fixed[fr] += slippage[:, None, :]               # broadcast joints
        if end < shadow_fixed.shape[0]:
            shadow_fixed[end:] += slippage[-1, None, :]        # carry forward

    return shadow_fixed


def compute_median_positions(filtered_rows):
    """
    Stage 4: compute per-step median hit position with skateboarding correction.

    Writes median_heel_positions.csv (one row per footstep) to OUT_CSV.

    Parameters
    ----------
    filtered_rows : list[dict]  output of filter_intersections()
    """
    print(f"\n[Stage 4] Computing per-step median heel positions …")

    print(f"  Loading {STEPS_CSV.name} …")
    df_steps = pd.read_csv(STEPS_CSV)
    n_frames = len(df_steps)
    print(f"  {n_frames:,} frames")

    # Convert filtered rows → DataFrame; coerce "nan" strings to actual NaN
    df_hits = pd.DataFrame(filtered_rows)
    df_hits["outlier"] = df_hits["outlier"].astype(str).str.lower() == "true"
    for col in ("hit_x", "hit_y", "hit_z"):
        df_hits[col] = pd.to_numeric(df_hits[col], errors="coerce")
    df_valid   = df_hits[~df_hits["outlier"]].copy()
    hit_lookup = df_valid.set_index("dji_frame")[["hit_x", "hit_y", "hit_z"]]
    print(f"  {len(df_hits):,} rows  ({len(df_valid):,} non-outlier)")

    # Build (n_frames, n_joints, 3) joint position array from CSV columns
    n_joints = len(JOINT_NAMES)
    points   = np.full((n_frames, n_joints, 3), np.nan, dtype=np.float64)
    for ji, name in enumerate(JOINT_NAMES):
        for ai, axis in enumerate(("x", "y", "z")):
            col = f"{name}_{axis}"
            if col in df_steps.columns:
                points[:, ji, ai] = df_steps[col].values

    # Detect heel strikes and apply skateboarding correction
    all_steps    = _detect_steps(df_steps)
    print(f"  {len(all_steps):,} heel strikes detected")
    shadow_fixed = _fix_skateboarding(points, JOINT_NAMES, all_steps)

    rheel_idx = JOINT_NAMES.index("RightHeel")
    lheel_idx = JOINT_NAMES.index("LeftHeel")

    # Compute per-step medians
    results  = []
    n_steps  = len(all_steps)

    for ss in range(n_steps - 1):
        row_start   = int(all_steps[ss,     0])
        row_end     = int(all_steps[ss + 1, 0])
        stance_leg  = int(all_steps[ss,     2])
        stance_str  = "RightFoot" if stance_leg == 1 else "LeftFoot"
        frame_start = int(all_steps[ss,     1])
        frame_end   = int(all_steps[ss + 1, 1])

        step_dji_frames = df_steps["dji_frame"].iloc[row_start:row_end].values
        step_hits       = hit_lookup.reindex(step_dji_frames).dropna()
        n_valid         = len(step_hits)

        if n_valid > 0:
            med_hit = step_hits.median()
            med_x, med_y, med_z = (med_hit["hit_x"],
                                   med_hit["hit_y"],
                                   med_hit["hit_z"])
        else:
            med_x = med_y = med_z = np.nan

        heel_idx = rheel_idx if stance_leg == 1 else lheel_idx
        sh_xyz   = shadow_fixed[row_start, heel_idx, :]

        results.append({
            "step_idx":      ss,
            "stance_foot":   stance_str,
            "frame_start":   frame_start,
            "frame_end":     frame_end,
            "n_frames":      row_end - row_start,
            "n_valid":       n_valid,
            "median_hit_x":  round(med_x, 6) if not np.isnan(med_x) else "",
            "median_hit_y":  round(med_y, 6) if not np.isnan(med_y) else "",
            "median_hit_z":  round(med_z, 6) if not np.isnan(med_z) else "",
            "shadow_heel_x": round(sh_xyz[0], 6),
            "shadow_heel_y": round(sh_xyz[1], 6),
            "shadow_heel_z": round(sh_xyz[2], 6),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\nDone — {len(out_df):,} steps written to {OUT_CSV.name}")
    print(f"  Steps with valid hits : {(out_df['n_valid'] > 0).sum():,}")
    print(f"  Steps with no hits    : {(out_df['n_valid'] == 0).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process heel data: ViTPose → ray-cast → filter → median positions."
    )
    parser.add_argument(
        "--pose_dir",
        type=Path
    )
    parser.add_argument(
        "--steps_csv",
        type=Path
    )
    parser.add_argument(
        "--ply_path",
        type=Path
    )
    parser.add_argument(
        "--json_path",
        type=Path
    )
    parser.add_argument(
        "--out_csv",
        type=Path
    )
    return parser.parse_args()


def main():
    global POSE_DIR, STEPS_CSV, PLY_PATH, JSON_PATH, OUT_CSV

    args = parse_args()
    POSE_DIR  = args.pose_dir
    STEPS_CSV = args.steps_csv
    PLY_PATH  = args.ply_path
    JSON_PATH = args.json_path
    OUT_CSV   = args.out_csv

    print("Paths:")
    print(f"  pose_dir  : {POSE_DIR}")
    print(f"  steps_csv : {STEPS_CSV}")
    print(f"  ply_path  : {PLY_PATH}")
    print(f"  json_path : {JSON_PATH}")
    print(f"  out_csv   : {OUT_CSV}")
    print()

    # Stage 1: Extract heel pixel positions + Shadow MoCap 3D positions
    heel_rows = extract_heel_data()

    # Stage 2: Ray-cast heel pixels through the 3D mesh
    intersection_rows = intersect_heels(heel_rows)

    # Stage 3: Remove spatial outliers with sliding-window MAD filter
    filtered_rows = filter_intersections(intersection_rows)

    # Stage 4: Compute per-step median positions → write out_csv
    compute_median_positions(filtered_rows)


if __name__ == "__main__":
    main()
