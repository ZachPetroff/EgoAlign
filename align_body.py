#!/usr/bin/env python3
"""
align_body.py
=============
Applies the windowed Umeyama transforms (saved by umeyama_align.py) to the
full Shadow MoCap skeleton, and precomputes Aria camera poses + gaze points in
DJI world space, so walk_viewer.py only needs to read one CSV.

Pipeline
--------
1. Load time_aligned_steps.csv  → raw joint positions (n_frames, n_joints, 3)
2. Re-apply fix_skateboarding   → corrected joint positions
3. Load per-step windowed transforms from umeyama_transform.npz
4. Interpolate transforms to per-frame using Slerp (rotation) +
   linear interp (translation, scale), anchored at each step's heel-strike frame
5. Apply per-frame transforms to every joint
6. Time-align each shadow frame to Aria via dji_aria_frame_matches.csv, look up
   device pose in aria/closed_loop_trajectory.csv, gaze pixel in aria/frames.csv,
   and transform everything into DJI world space using alignment_transform.npz
7. Cast each gaze ray against the DJI mesh (Open3D BVH → trimesh fallback) to
   find the true 3-D gaze intersection point — Aria depth is NOT used
8. Write detected_steps_aligned.csv with:
     {joint}_aligned_x/y/z  columns  (skeleton, for walk_viewer.py)
     cam_pos_x/y/z           camera position in DJI world (= Head joint)
     cam_rot_r00…r22         3×3 camera rotation (row-major, 9 columns)
     gaze_x/y/z              3-D mesh intersection point (NaN if no hit)
     gaze_valid              1 if ray hit the mesh, else 0

Usage:
    python3 align_body.py \\
        --steps-csv time_aligned_steps.csv \\
        --npz       umeyama_transform.npz \\
        --out-csv   detected_steps_aligned.csv \\
        [--dji-aria-matches dji_aria_frame_matches.csv] \\
        [--closed-loop-traj aria/closed_loop_trajectory.csv] \\
        [--frames-csv       aria/frames.csv] \\
        [--alignment-npz    alignment_transform.npz] \\
        [--ply              dji/recon_1.ply]

The three required arguments (--steps-csv, --npz, --out-csv) must always be
supplied.  The five Aria/mesh arguments are optional; if any are omitted the
Aria camera and gaze columns are skipped (a warning is printed).
"""

import argparse
import csv
import os
import sys
import time as _time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

GAZE_MAX_DIST = 200.0   # metres — rays beyond this count as misses

# ── Aria camera intrinsics (from CameraCalibration.txt) ──────────────────────
ARIA_FX = 610.63
ARIA_FY = 610.63
ARIA_CX = 653.5
ARIA_CY = 653.5

# Extrinsics: T_Device_Camera — pose of the camera origin in device frame
_T_DC_T = np.array([-0.00431336, -0.0116055, -0.00499766])
_T_DC_Q = {"x": 0.335775, "y": 0.0344963, "z": 0.0377337, "w": 0.940554}

# All joints present in time_aligned_steps.csv (order must match the array axis)
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


# ── Math helpers ─────────────────────────────────────────────────────────────

def quat_to_rot(q: dict) -> np.ndarray:
    """Quaternion dict {'x','y','z','w'} → 3×3 rotation matrix."""
    qx, qy, qz, qw = float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def _nearest(arr: np.ndarray, val: int) -> int:
    """Index of nearest value in a sorted int64 array."""
    idx = int(np.searchsorted(arr, val))
    idx = min(idx, len(arr) - 1)
    if idx > 0 and (arr[idx - 1] - val) ** 2 < (arr[idx] - val) ** 2:
        idx -= 1
    return idx


# ── PLY geometry loader ───────────────────────────────────────────────────────

def load_ply_geometry(path: str) -> tuple:
    """
    Fast binary PLY parser — extracts only vertex positions and triangle faces.
    Returns (verts, faces): float32 (N,3) and int32 (M,3).
    """
    print(f"  Parsing PLY: {path} …", end=" ", flush=True)
    t0 = _time.time()
    with open(path, "rb") as f:
        n_verts = n_faces = 0
        vert_props = []
        current_element = None
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            if line == "end_header":
                break
            tokens = line.split()
            if line.startswith("element vertex"):
                n_verts = int(tokens[-1]); current_element = "vertex"
            elif line.startswith("element face"):
                n_faces = int(tokens[-1]); current_element = "face"
            elif line.startswith("property") and current_element == "vertex":
                if tokens[1] != "list":
                    vert_props.append(tokens[-1])
        n_vp = len(vert_props)
        raw_v = f.read(n_verts * n_vp * 4)
        verts = np.frombuffer(raw_v, dtype="<f4").reshape(n_verts, n_vp)[:, :3].copy()
        face_dtype = np.dtype([("n", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")])
        raw_f = f.read(n_faces * face_dtype.itemsize)
        fa = np.frombuffer(raw_f, dtype=face_dtype)
        if not np.all(fa["n"] == 3):
            fa = fa[fa["n"] == 3]
        faces = np.stack([fa["i0"], fa["i1"], fa["i2"]], axis=1).astype(np.int32)
    print(f"{n_verts:,} verts  {len(faces):,} faces  ({_time.time()-t0:.1f}s)", flush=True)
    return verts, faces


# ── Raycasting backends ───────────────────────────────────────────────────────

def _build_scene_open3d(verts, faces):
    try:
        import open3d as o3d
        import open3d.core as o3c
        if not hasattr(o3d, "t"):
            return None, "open3d.t unavailable"
        print("  Building Open3D BVH …", end=" ", flush=True)
        t0 = _time.time()
        mesh_t = o3d.t.geometry.TriangleMesh()
        mesh_t.vertex.positions = o3c.Tensor(verts.astype(np.float32))
        mesh_t.triangle.indices = o3c.Tensor(faces.astype(np.int32))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        print(f"{_time.time()-t0:.1f}s", flush=True)
        return scene, "open3d"
    except Exception as e:
        return None, str(e)


def _build_scene_trimesh(verts, faces):
    try:
        import trimesh
        print("  Building trimesh scene …", end=" ", flush=True)
        t0 = _time.time()
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        try:
            import pyembree  # noqa: F401
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
            backend = "trimesh+pyembree"
        except ImportError:
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
            backend = "trimesh"
        print(f"{_time.time()-t0:.1f}s", flush=True)
        return intersector, backend
    except Exception as e:
        return None, str(e)


def _cast_open3d(scene, origins, directions, max_dist):
    import open3d.core as o3c
    rays_np = np.concatenate([origins.astype(np.float32),
                               directions.astype(np.float32)], axis=1)
    t_hit = scene.cast_rays(o3c.Tensor(rays_np, dtype=o3c.float32))["t_hit"].numpy()
    hit = np.isfinite(t_hit) & (t_hit <= max_dist)
    t_hit[~hit] = np.inf
    return hit, t_hit.astype(np.float64)


def _cast_trimesh(intersector, origins, directions, max_dist):
    locs, index_ray, _ = intersector.intersects_location(
        origins, directions, multiple_hits=False)
    N = len(origins)
    hit = np.zeros(N, dtype=bool)
    dist = np.full(N, np.inf)
    if len(index_ray):
        d = np.linalg.norm(locs - origins[index_ray], axis=1)
        for j in range(len(index_ray)):
            i = index_ray[j]
            if d[j] <= max_dist and d[j] < dist[i]:
                dist[i] = d[j]; hit[i] = True
    return hit, dist


# ── fix_skateboarding (same as median_heel_positions.py) ─────────────────────

def fix_skateboarding(points, parts, all_steps):
    """
    Correct foot-slippage artifacts.
    points    : (n_frames, n_joints, 3)
    parts     : list[str] – joint names for axis 1
    all_steps : (N, 3) int – columns [row_idx, dji_frame, stance_leg]
    Returns corrected copy of points.
    """
    shadow_fixed = points.copy()
    rheel_idx = parts.index("RightHeel")
    lheel_idx = parts.index("LeftHeel")
    hs         = all_steps[:, 0].astype(int)
    stance_leg = all_steps[:, 2].astype(int)

    for ss in range(len(hs) - 1):
        start = hs[ss];  end = hs[ss + 1]
        fr    = np.arange(start, end)
        hidx  = rheel_idx if stance_leg[ss] == 1 else lheel_idx
        foothold  = shadow_fixed[start, hidx, :]
        curr_heel = shadow_fixed[fr,    hidx, :]
        slippage  = foothold[None, :] - curr_heel        # (len(fr), 3)
        shadow_fixed[fr] += slippage[:, None, :]
        if end < shadow_fixed.shape[0]:
            shadow_fixed[end:] += slippage[-1, None, :]

    return shadow_fixed


# ── Step detection (same logic as median_heel_positions.py) ──────────────────

def detect_steps(df):
    foot_col    = df["max_pressure_foot"].values
    change_mask = np.concatenate([[True], foot_col[1:] != foot_col[:-1]])
    change_rows = np.where(change_mask)[0]
    steps = []
    for ri in change_rows:
        dji_frame  = int(df["dji_frame"].iloc[ri])
        stance_leg = 1 if foot_col[ri] == "RightFoot" else 0
        steps.append([ri, dji_frame, stance_leg])
    return np.array(steps, dtype=int)


# ── Per-frame transform interpolation ────────────────────────────────────────

def interpolate_transforms(all_steps, sm_quats, sm_trans, sm_scales, n_frames):
    """
    Interpolate per-step transforms to per-frame.

    Anchor frames are the heel-strike rows (all_steps[:, 0]).
    Rotation : Slerp
    Translation / scale : linear interp1d, fill_value='extrapolate'

    Returns
    -------
    frame_quats  : (n_frames, 4)
    frame_trans  : (n_frames, 3)
    frame_scales : (n_frames,)
    """
    n_steps       = len(sm_quats)
    anchor_frames = all_steps[:n_steps, 0].astype(float)   # shape (N_steps,)
    all_frames    = np.arange(n_frames, dtype=float)

    # ── Rotation via Slerp ────────────────────────────────────────────────────
    rotations = Rotation.from_quat(sm_quats)         # (N_steps,)
    slerp     = Slerp(anchor_frames, rotations)

    # Slerp only interpolates inside [anchor_frames[0], anchor_frames[-1]].
    # Clamp out-of-range frames to boundary values to extrapolate constantly.
    clamped = np.clip(all_frames, anchor_frames[0], anchor_frames[-1])
    frame_rots  = slerp(clamped)
    frame_quats = frame_rots.as_quat()               # (n_frames, 4)

    # ── Translation ───────────────────────────────────────────────────────────
    interp_t = interp1d(anchor_frames, sm_trans, axis=0,
                        kind="linear", bounds_error=False,
                        fill_value=(sm_trans[0], sm_trans[-1]))
    frame_trans = interp_t(all_frames)               # (n_frames, 3)

    # ── Scale ─────────────────────────────────────────────────────────────────
    interp_s = interp1d(anchor_frames, sm_scales,
                        kind="linear", bounds_error=False,
                        fill_value=(sm_scales[0], sm_scales[-1]))
    frame_scales = interp_s(all_frames)              # (n_frames,)

    return frame_quats, frame_trans, frame_scales


# ── Apply per-frame transforms to all joints ─────────────────────────────────

def apply_per_frame(points_fixed, frame_quats, frame_trans, frame_scales):
    """
    points_fixed  : (n_frames, n_joints, 3)
    Returns aligned array of same shape.
    """
    n_frames, n_joints, _ = points_fixed.shape
    aligned = np.empty_like(points_fixed)

    # Batch over frames
    for fi in range(n_frames):
        R = Rotation.from_quat(frame_quats[fi]).as_matrix()   # (3, 3)
        c = frame_scales[fi]
        t = frame_trans[fi]
        # (c * R @ pts.T).T + t  for all joints at once
        aligned[fi] = (c * (R @ points_fixed[fi].T)).T + t    # (n_joints, 3)

    return aligned


# ── Aria time-alignment + world-space transform ───────────────────────────────

def compute_aria_data(df: pd.DataFrame, n_frames: int,
                      head_positions: np.ndarray,
                      alignment_npz_path: str | None,
                      dji_aria_matches_path: str | None,
                      closed_loop_traj_path: str | None,
                      frames_csv_path: str | None,
                      ply_path: str | None) -> dict | None:
    """
    For every shadow frame, look up the Aria device pose and gaze data,
    transform both into DJI world space, and return a dict of column arrays.

    Camera position is taken directly from the aligned skeleton Head joint
    (head_positions, shape (n_frames, 3) in DJI world space).
    Camera rotation is derived from the Aria trajectory as usual.

    Time chain per frame i
    ----------------------
    df["dji_frame"][i]
        → aria_utc_timestamp_ns  (via dji_aria_frame_matches.csv)
        → nearest row in closed_loop_trajectory.csv  (by utc_timestamp_ns)
            → device rotation + tracking_timestamp_us
            → nearest row in frames.csv → gaze pixel (depth ignored)
        → apply T_Device_Camera extrinsics  → camera rotation in Aria world
        → apply inv(alignment_transform["T_total"])  → DJI world space
        → cast unbounded ray against DJI mesh → 3-D intersection point

    Returns None if any required file is missing.
    """
    required = {
        "alignment_npz":      alignment_npz_path,
        "dji_aria_matches":   dji_aria_matches_path,
        "closed_loop_traj":   closed_loop_traj_path,
        "frames_csv":         frames_csv_path,
    }
    for label, p in required.items():
        if p is None or not os.path.exists(p):
            display = p if p is not None else f"<{label} not provided>"
            print(f"  WARNING: {display!r} not found – skipping Aria columns.")
            return None

    # ── Alignment transform ───────────────────────────────────────────────────
    npz           = np.load(alignment_npz_path)
    T_dji_to_aria = npz["T_total"]
    T_aria_to_dji = np.linalg.inv(T_dji_to_aria)
    R_a2d = T_aria_to_dji[:3, :3]
    t_a2d = T_aria_to_dji[:3,  3]
    print(f"  Alignment transform loaded  "
          f"(inv-cond={np.linalg.cond(T_dji_to_aria):.2f})")

    # ── Camera extrinsics ─────────────────────────────────────────────────────
    R_dc = quat_to_rot(_T_DC_Q)
    t_dc = _T_DC_T

    # ── dji_frame → aria_utc_timestamp_ns ────────────────────────────────────
    with open(dji_aria_matches_path) as f:
        matches_rows = list(csv.DictReader(f))
    max_dji = max(int(r["dji_frame"]) for r in matches_rows)
    aria_ts_by_dji = np.zeros(max_dji + 1, dtype=np.int64)
    for r in matches_rows:
        aria_ts_by_dji[int(r["dji_frame"])] = int(r["aria_utc_timestamp_ns"])
    dji_frames = df["dji_frame"].values.astype(np.int32)

    # ── Load closed_loop_trajectory.csv ──────────────────────────────────────
    print(f"  Loading closed_loop_trajectory.csv …", end=" ", flush=True)
    with open(closed_loop_traj_path) as f:
        traj_rows = list(csv.DictReader(f))
    traj_utc = np.array([int(e["utc_timestamp_ns"]) for e in traj_rows], dtype=np.int64)
    print(f"{len(traj_rows)} entries")

    # ── Load frames.csv (gaze) ────────────────────────────────────────────────
    print(f"  Loading gaze data …", end=" ", flush=True)
    with open(frames_csv_path) as f:
        gaze_rows = list(csv.DictReader(f))
    gaze_track_ns = np.array(
        [int(r["capture_timestamp_ns"]) for r in gaze_rows], dtype=np.int64
    )
    gaze_u = np.full(len(gaze_rows), np.nan)
    gaze_v = np.full(len(gaze_rows), np.nan)
    for k, r in enumerate(gaze_rows):
        if r["has_gaze"] == "1":
            gaze_u[k] = float(r["gaze_u_px"])
            gaze_v[k] = float(r["gaze_v_px"])
    print(f"{int(np.sum(~np.isnan(gaze_u)))}/{len(gaze_rows)} gaze frames (depth ignored)")

    # ── Per-frame: camera pose + gaze ray direction ───────────────────────────
    print(f"  Computing {n_frames} camera poses + gaze ray directions …",
          end=" ", flush=True)
    cam_pos      = np.empty((n_frames, 3),    dtype=np.float64)
    cam_pos_aria = np.empty((n_frames, 3),    dtype=np.float64)  # true Aria camera position
    cam_rot      = np.empty((n_frames, 3, 3), dtype=np.float64)
    ray_dir      = np.full((n_frames, 3), np.nan, dtype=np.float64)
    has_gaze     = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        ts_target = aria_ts_by_dji[dji_frames[i]]

        # Nearest trajectory row by UTC timestamp → device pose
        idx_traj = _nearest(traj_utc, ts_target)
        entry    = traj_rows[idx_traj]

        p_wd = np.array([float(entry["tx_world_device"]),
                         float(entry["ty_world_device"]),
                         float(entry["tz_world_device"])], dtype=np.float64)
        R_wd = quat_to_rot({"x": entry["qx_world_device"],
                            "y": entry["qy_world_device"],
                            "z": entry["qz_world_device"],
                            "w": entry["qw_world_device"]})

        # True Aria camera position in DJI world (used as ray origin)
        R_cam_aria = R_wd @ R_dc
        p_cam_aria = R_wd @ t_dc + p_wd          # camera origin in Aria world
        cam_pos_aria[i] = R_a2d @ p_cam_aria + t_a2d

        # Display position = Head joint; rotation from trajectory
        cam_pos[i] = head_positions[i]
        cam_rot[i] = R_a2d @ R_cam_aria

        # Gaze ray direction in DJI world (Aria depth ignored)
        track_ns = int(entry["tracking_timestamp_us"]) * 1000
        idx_gaze = _nearest(gaze_track_ns, track_ns)
        if not np.isnan(gaze_u[idx_gaze]):
            u, v = gaze_u[idx_gaze], gaze_v[idx_gaze]
            dir_cam = np.array([(u - ARIA_CX) / ARIA_FX,
                                (v - ARIA_CY) / ARIA_FY,
                                1.0], dtype=np.float64)
            dir_cam /= np.linalg.norm(dir_cam)
            ray_dir[i]  = cam_rot[i] @ dir_cam
            has_gaze[i] = True

    n_gaze = int(has_gaze.sum())
    print(f"done  ({n_gaze}/{n_frames} have gaze)", flush=True)

    # ── Mesh raycasting ───────────────────────────────────────────────────────
    print(f"  Loading mesh for raycasting …")
    gaze_xyz   = np.full((n_frames, 3), np.nan, dtype=np.float64)
    gaze_valid = np.zeros(n_frames, dtype=bool)

    if ply_path is None or not os.path.exists(ply_path):
        display_ply = ply_path if ply_path is not None else "<ply not provided>"
        print(f"  WARNING: {display_ply!r} not found – gaze intersections skipped.")
    else:
        verts, faces = load_ply_geometry(ply_path)

        scene, backend = _build_scene_open3d(verts, faces)
        if scene is None:
            print(f"  Open3D unavailable ({backend}), trying trimesh …")
            scene, backend = _build_scene_trimesh(verts, faces)
        if scene is None:
            print(f"  WARNING: no raycasting backend available – gaze intersections skipped.")
        else:
            print(f"  Backend: {backend}")
            gaze_idx   = np.where(has_gaze)[0]
            origins    = cam_pos_aria[gaze_idx]   # true Aria camera positions
            directions = ray_dir[gaze_idx]
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            directions = directions / np.where(norms > 0, norms, 1.0)

            print(f"  Casting {n_gaze} gaze rays …", end=" ", flush=True)
            t0 = _time.time()
            if backend == "open3d":
                hit_mask, hit_dist = _cast_open3d(scene, origins, directions, GAZE_MAX_DIST)
            else:
                hit_mask, hit_dist = _cast_trimesh(scene, origins, directions, GAZE_MAX_DIST)
            print(f"{_time.time()-t0:.1f}s", flush=True)

            hit_mask_full = np.zeros(n_frames, dtype=bool)
            hit_dist_full = np.full(n_frames, np.inf)
            hit_mask_full[gaze_idx] = hit_mask
            hit_dist_full[gaze_idx] = hit_dist

            gaze_valid = hit_mask_full
            # Hit point = true Aria ray origin + direction * distance
            gaze_xyz[gaze_valid] = (cam_pos_aria[gaze_valid]
                                    + ray_dir[gaze_valid]
                                    * hit_dist_full[gaze_valid, None])
            n_hit = int(gaze_valid.sum())
            print(f"  Hits: {n_hit}/{n_gaze} gaze rays  ({100*n_hit/max(n_gaze,1):.1f}%)")

    # ── Pack into column dict ─────────────────────────────────────────────────
    cols: dict = {}
    cols["cam_pos_x"] = cam_pos[:, 0]
    cols["cam_pos_y"] = cam_pos[:, 1]
    cols["cam_pos_z"] = cam_pos[:, 2]
    for ri in range(3):
        for ci in range(3):
            cols[f"cam_rot_r{ri}{ci}"] = cam_rot[:, ri, ci]
    cols["gaze_x"]     = gaze_xyz[:, 0]
    cols["gaze_y"]     = gaze_xyz[:, 1]
    cols["gaze_z"]     = gaze_xyz[:, 2]
    cols["gaze_valid"] = gaze_valid.astype(np.int8)
    return cols


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Apply windowed Umeyama transforms to the Shadow MoCap "
                    "skeleton and optionally precompute Aria camera poses / "
                    "gaze points in DJI world space."
    )

    # ── Required paths ────────────────────────────────────────────────────────
    p.add_argument("--steps-csv", required=True,
                   help="Input CSV with time-aligned step/joint data "
                        "(e.g. time_aligned_steps.csv).")
    p.add_argument("--npz", required=True,
                   help="NPZ file with per-step windowed Umeyama transforms "
                        "(e.g. umeyama_transform.npz).")
    p.add_argument("--out-csv", required=True,
                   help="Path for the output aligned CSV "
                        "(e.g. detected_steps_aligned.csv).")

    # ── Optional Aria / mesh paths ────────────────────────────────────────────
    p.add_argument("--dji-aria-matches",
                   help="CSV mapping DJI frames to Aria UTC timestamps "
                        "(e.g. dji_aria_frame_matches.csv).")
    p.add_argument("--closed-loop-traj",
                   help="Aria closed-loop trajectory CSV "
                        "(e.g. aria/closed_loop_trajectory.csv).")
    p.add_argument("--frames-csv",
                   help="Aria frames CSV with per-frame gaze pixels "
                        "(e.g. aria/frames.csv).")
    p.add_argument("--alignment-npz",
                   help="NPZ with the DJI↔Aria alignment transform "
                        "(e.g. alignment_transform.npz).")
    p.add_argument("--ply",
                   help="DJI reconstruction PLY mesh used for gaze raycasting "
                        "(e.g. dji/recon_1.ply).")

    return p.parse_args()


def main():
    args = parse_args()

    steps_csv             = args.steps_csv
    npz_path              = args.npz
    out_csv               = args.out_csv
    dji_aria_matches_path = args.dji_aria_matches
    closed_loop_traj_path = args.closed_loop_traj
    frames_csv_path       = args.frames_csv
    alignment_npz_path    = args.alignment_npz
    ply_path              = args.ply

    # ── Load raw skeleton ────────────────────────────────────────────────────
    df = pd.read_csv(steps_csv)
    n_frames  = len(df)
    n_joints  = len(JOINT_NAMES)
    print(f"  {n_frames} frames, {n_joints} joints")

    print("Building joint position array …")
    points = np.full((n_frames, n_joints, 3), np.nan)
    for ji, name in enumerate(JOINT_NAMES):
        for ai, axis in enumerate(("x", "y", "z")):
            col = f"{name}_{axis}"
            if col in df.columns:
                points[:, ji, ai] = df[col].values

    # ── Step detection ───────────────────────────────────────────────────────
    print("Detecting steps …")
    all_steps = detect_steps(df)
    print(f"  {len(all_steps)} heel strikes")

    # ── Skateboarding correction ─────────────────────────────────────────────
    print("Applying fix_skateboarding …")
    points_fixed = fix_skateboarding(points, JOINT_NAMES, all_steps)

    # ── Load windowed transforms ─────────────────────────────────────────────
    npz = np.load(npz_path)
    sm_quats  = npz["sm_quats"]    # (n_steps, 4)
    sm_trans  = npz["sm_trans"]    # (n_steps, 3)
    sm_scales = npz["sm_scales"]   # (n_steps,)

    # Describe what was loaded — NPZ format changed when adaptive mode was added:
    # old format has 'window_size' (scalar); new format has 'adaptive_windows' (array).
    smooth_sigma = float(npz["smooth_sigma"]) if "smooth_sigma" in npz else float("nan")
    if "adaptive_windows" in npz:
        aw = npz["adaptive_windows"]
        mode_str = (f"adaptive  window range=[{aw.min()}, {aw.max()}]  "
                    f"σ={smooth_sigma}")
    elif "window_size" in npz:
        mode_str = f"fixed  W={int(npz['window_size'])}  σ={smooth_sigma}"
    else:
        mode_str = f"σ={smooth_sigma}"
    print(f"  {len(sm_quats)} per-step transforms  ({mode_str})")

    # ── Interpolate transforms to per-frame ──────────────────────────────────
    print("Interpolating transforms to per-frame …")
    frame_quats, frame_trans, frame_scales = interpolate_transforms(
        all_steps, sm_quats, sm_trans, sm_scales, n_frames
    )

    # ── Apply to all joints ──────────────────────────────────────────────────
    print("Applying per-frame transforms to all joints …")
    aligned = apply_per_frame(points_fixed, frame_quats, frame_trans, frame_scales)
    print("  Done.")

    # ── Aria camera poses + gaze in DJI world ───────────────────────────────
    print("Computing Aria camera poses + gaze …")
    head_idx      = JOINT_NAMES.index("Head")
    head_positions = aligned[:, head_idx, :]   # (n_frames, 3) in DJI world
    aria_cols = compute_aria_data(
        df, n_frames, head_positions,
        alignment_npz_path=alignment_npz_path,
        dji_aria_matches_path=dji_aria_matches_path,
        closed_loop_traj_path=closed_loop_traj_path,
        frames_csv_path=frames_csv_path,
        ply_path=ply_path,
    )

    # ── Build output DataFrame ───────────────────────────────────────────────
    out = pd.DataFrame()
    out["time"]              = df["time"]
    out["max_pressure_foot"] = df["max_pressure_foot"]

    for ji, name in enumerate(JOINT_NAMES):
        out[f"{name}_aligned_x"] = aligned[:, ji, 0]
        out[f"{name}_aligned_y"] = aligned[:, ji, 1]
        out[f"{name}_aligned_z"] = aligned[:, ji, 2]

    if aria_cols is not None:
        for col, arr in aria_cols.items():
            out[col] = arr

    out.to_csv(out_csv, index=False)
    aria_note = " (with Aria columns)" if aria_cols is not None else " (no Aria columns)"
    print(f"\nWrote {out_csv}{aria_note}")
    print("Run:  python3 walk_viewer.py")


if __name__ == "__main__":
    main()