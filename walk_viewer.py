#!/usr/bin/env python3
"""
walk_viewer.py
==============
Interactive 3D viewer: aligned Shadow MoCap skeleton walking through the
drone photogrammetry reconstruction, with a time-aligned Aria camera frustum
and 3D gaze ray.

Aria camera poses and gaze points are precomputed by align_body.py and stored
as columns in detected_steps_aligned.csv — this viewer just reads them.

Controls
--------
  Mouse left-drag   – rotate
  Mouse right-drag  – pan
  Scroll wheel      – zoom
  Space             – pause / resume animation
  Left / Right      – step one frame at a time (while paused)
  R                 – restart from frame 0
  [ / ]             – slow down / speed up playback (0.25× – 4×)
  F                 – toggle Aria frustum + camera trajectory
  G                 – toggle gaze ray
  Q  or  Esc        – quit

Run
---
  python3 align_body.py   # (re-)generates detected_steps_aligned.csv
  python3 walk_viewer.py
"""

import csv
import os
import time

import numpy as np
import open3d as o3d

PLY_PATH  = "dji/recon_1.ply"
CSV_PATH  = "detected_steps_aligned.csv"
FRAMES_DIR = "walk_frames"   # parent folder; each recording gets a timestamped sub-folder

N_MESH_PTS   = 2_000_000
FRAME_STRIDE = 1
TARGET_FPS   = 30

# ── Aria camera intrinsics (from CameraCalibration.txt) ───────────────────────
ARIA_FX = 610.63
ARIA_FY = 610.63
ARIA_CX = 653.5
ARIA_CY = 653.5

# Display parameters
FRUSTUM_NEAR     = 0.05
FRUSTUM_FAR      = 0.30
FRUSTUM_COLOR    = [1.0, 0.9, 0.1]    # yellow
ARIA_TRAJ_COLOR  = [0.5, 0.5, 0.15]   # dim yellow
GAZE_COLOR       = [1.0, 0.2, 0.9]    # magenta
GAZE_CROSS_HALF  = 0.04               # half-size of the cross at the gaze hit (metres)

# ── Gaze heatmap ──────────────────────────────────────────────────────────────
HEATMAP_SIGMA  = 0.5    # Gaussian kernel σ in metres
HEATMAP_RADIUS = 1.5    # search radius (= 3σ); points beyond this are unaffected

# ── Skeleton ──────────────────────────────────────────────────────────────────
BONES = [
    ("Hips","SpineLow"),("SpineLow","SpineMid"),("SpineMid","Chest"),
    ("Chest","Neck"),("Neck","Head"),("Head","HeadEnd"),
    ("Hips","RightThigh"),("RightThigh","RightLeg"),
    ("RightLeg","RightFoot"),("RightFoot","RightToe"),("RightFoot","RightHeel"),
    ("Hips","LeftThigh"),("LeftThigh","LeftLeg"),
    ("LeftLeg","LeftFoot"),("LeftFoot","LeftToe"),("LeftFoot","LeftHeel"),
    ("Chest","RightShoulder"),("RightShoulder","RightArm"),
    ("RightArm","RightForearm"),("RightForearm","RightHand"),
    ("Chest","LeftShoulder"),("LeftShoulder","LeftArm"),
    ("LeftArm","LeftForearm"),("LeftForearm","LeftHand"),
]

JOINT_NAMES = [
    "Body","Hips","SpineLow","SpineMid","Chest","Neck","Head","HeadEnd",
    "RightThigh","RightLeg","RightFoot","RightToe","RightToeEnd","RightHeel",
    "LeftThigh","LeftLeg","LeftFoot","LeftToe","LeftToeEnd","LeftHeel",
    "RightShoulder","RightArm","RightForearm","RightHand",
    "LeftShoulder","LeftArm","LeftForearm","LeftHand",
]

def bone_color(p, c):
    if p.startswith("Right") or c.startswith("Right"): return [1.0, 0.35, 0.35]
    if p.startswith("Left")  or c.startswith("Left"):  return [0.35, 0.85, 1.0]
    return [0.95, 0.95, 0.95]

BONE_COLORS = np.array([bone_color(p, c) for p, c in BONES])


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ply_sample(path, n_sample):
    print(f"Loading PLY ({n_sample:,} sample points) …", flush=True)
    with open(path, "rb") as f:
        lines, in_vert, n_verts, props = [], False, 0, []
        while True:
            ln = f.readline().decode("utf-8", errors="replace").strip()
            lines.append(ln)
            if ln == "end_header": break
        for ln in lines:
            toks = ln.split()
            if ln.startswith("element vertex"):
                n_verts = int(toks[-1]); in_vert = True
            elif ln.startswith("element") and not ln.startswith("element vertex"):
                in_vert = False
            elif ln.startswith("property") and in_vert:
                props.append(toks[-1])
        raw = f.read(n_verts * len(props) * 4)
    arr = np.frombuffer(raw, dtype="<f4").reshape(n_verts, len(props))
    idx = np.random.choice(n_verts, min(n_sample, n_verts), replace=False)
    xyz = arr[idx, :3].astype(np.float64)
    rgb = np.clip(arr[idx, 3:6].astype(np.float64), 0, 1)
    if rgb.max() > 1.01: rgb /= 255.0
    print(f"  {len(xyz):,} points loaded", flush=True)
    return xyz, rgb


def _fnan(s: str) -> float:
    """Parse a CSV float that may be empty or 'nan'."""
    return float(s) if s and s.lower() != "nan" else float("nan")


def load_frames(path):
    """
    Load detected_steps_aligned.csv.

    Always returns skeleton joint arrays and metadata.
    If Aria columns are present (written by align_body.py), also returns:
        cam_pos   (N, 3)    camera position in DJI world
        cam_rot   (N, 3, 3) camera rotation in DJI world
        gaze_pt   (N, 3)    mesh-intersection hit point (NaN if no hit)
        gaze_valid (N,) bool  True when ray hit the mesh
    """
    print(f"Loading body frames …", flush=True)
    rows = list(csv.DictReader(open(path)))
    n = len(rows)
    data = {}
    for j in JOINT_NAMES:
        try:
            data[j] = np.array([[float(r[f"{j}_aligned_x"]),
                                  float(r[f"{j}_aligned_y"]),
                                  float(r[f"{j}_aligned_z"])] for r in rows],
                                dtype=np.float64)
        except KeyError:
            pass
    data["time"] = np.array([float(r["time"]) for r in rows])
    data["foot"] = [r["max_pressure_foot"] for r in rows]

    # Optional Aria columns (present only if align_body.py wrote them)
    if "cam_pos_x" in rows[0]:
        data["cam_pos"] = np.array(
            [[float(r["cam_pos_x"]), float(r["cam_pos_y"]), float(r["cam_pos_z"])]
             for r in rows], dtype=np.float64)
        data["cam_rot"] = np.array(
            [[[float(r[f"cam_rot_r{ri}{ci}"]) for ci in range(3)] for ri in range(3)]
             for r in rows], dtype=np.float64)
        data["gaze_pt"] = np.array(
            [[_fnan(r["gaze_x"]), _fnan(r["gaze_y"]), _fnan(r["gaze_z"])]
             for r in rows], dtype=np.float64)
        data["gaze_valid"] = np.array([r["gaze_valid"] == "1" for r in rows])
        n_hit = int(data["gaze_valid"].sum())
        print(f"  Aria overlay: {n_hit}/{n} mesh-intersection gaze frames", flush=True)

    print(f"  {n} frames  (t={rows[0]['time']}s – {rows[-1]['time']}s)", flush=True)
    return data, n


# ── Frustum geometry ──────────────────────────────────────────────────────────
#
# Camera-local frame: +Z forward, +X right, +Y down  (standard pinhole).
# 9 vertices: apex(0), near TL/TR/BR/BL(1-4), far TL/TR/BR/BL(5-8)
# 12 edges  : 4 apex→far, 4 near quad, 4 far quad

_TAN_HX = ARIA_CX / ARIA_FX   # ≈ 1.070
_TAN_HY = ARIA_CY / ARIA_FY   # ≈ 1.070

def _build_frustum_template() -> np.ndarray:
    n, f = FRUSTUM_NEAR, FRUSTUM_FAR
    wn, hn = n * _TAN_HX, n * _TAN_HY
    wf, hf = f * _TAN_HX, f * _TAN_HY
    return np.array([
        [ 0,   0,  0],   # 0 apex
        [-wn, -hn,  n],  # 1 near TL
        [ wn, -hn,  n],  # 2 near TR
        [ wn,  hn,  n],  # 3 near BR
        [-wn,  hn,  n],  # 4 near BL
        [-wf, -hf,  f],  # 5 far TL
        [ wf, -hf,  f],  # 6 far TR
        [ wf,  hf,  f],  # 7 far BR
        [-wf,  hf,  f],  # 8 far BL
    ], dtype=np.float64)

FRUSTUM_TEMPLATE = _build_frustum_template()
FRUSTUM_EDGES = np.array([
    [0, 5], [0, 6], [0, 7], [0, 8],  # apex → far corners
    [1, 2], [2, 3], [3, 4], [4, 1],  # near quad
    [5, 6], [6, 7], [7, 8], [8, 5],  # far  quad
], dtype=np.int32)


def make_frustum_ls() -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(FRUSTUM_TEMPLATE.copy())
    ls.lines  = o3d.utility.Vector2iVector(FRUSTUM_EDGES)
    ls.paint_uniform_color(FRUSTUM_COLOR)
    return ls


def update_frustum_ls(ls, pos, rot):
    """Transform frustum template to the current camera pose (in-place)."""
    world_pts = (rot @ FRUSTUM_TEMPLATE.T).T + pos
    ls.points = o3d.utility.Vector3dVector(world_pts)


# ── Gaze ray geometry ─────────────────────────────────────────────────────────
#
# Layout (8 points, 4 lines):
#   0  camera position
#   1  gaze hit point                     → line (0,1): the ray
#   2,3  hit ± (half, 0, 0) in world X    → line (2,3): cross arm X
#   4,5  hit ± (0, half, 0) in world Y    → line (4,5): cross arm Y
#   6,7  hit ± (0, 0, half) in world Z    → line (6,7): cross arm Z

GAZE_EDGES = np.array([
    [0, 1],   # ray
    [2, 3], [4, 5], [6, 7],  # cross at hit
], dtype=np.int32)

_GAZE_DUMMY = np.zeros((8, 3), dtype=np.float64)  # collapsed to origin when hidden


def make_gaze_ls() -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(_GAZE_DUMMY.copy())
    ls.lines  = o3d.utility.Vector2iVector(GAZE_EDGES)
    ls.paint_uniform_color(GAZE_COLOR)
    return ls


def update_gaze_ls(ls, cam_pos, gaze_pt, valid):
    """
    Update the gaze LineSet in-place.

    If not valid, collapse all points to cam_pos so nothing is drawn.
    """
    if not valid:
        pts = np.tile(cam_pos, (8, 1))
    else:
        h = GAZE_CROSS_HALF
        pts = np.array([
            cam_pos,
            gaze_pt,
            gaze_pt + [h, 0, 0],   gaze_pt - [h, 0, 0],
            gaze_pt + [0, h, 0],   gaze_pt - [0, h, 0],
            gaze_pt + [0, 0, h],   gaze_pt - [0, 0, h],
        ], dtype=np.float64)
        ls.paint_uniform_color(GAZE_COLOR)
    ls.points = o3d.utility.Vector3dVector(pts)


# ── Aria camera trajectory ghost ──────────────────────────────────────────────

def make_aria_traj_ls(cam_pos_dji):
    n  = len(cam_pos_dji)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(cam_pos_dji)
    idx = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    ls.lines  = o3d.utility.Vector2iVector(idx)
    ls.paint_uniform_color(ARIA_TRAJ_COLOR)
    return ls


# ── Skeleton geometry ─────────────────────────────────────────────────────────

def make_skeleton_geometry(body, fi):
    joints = {j: body[j][fi] for j in JOINT_NAMES if j in body}
    pts, lines, colors = [], [], []
    for bi, (p, c) in enumerate(BONES):
        if p in joints and c in joints:
            i0 = len(pts); pts.append(joints[p])
            i1 = len(pts); pts.append(joints[c])
            lines.append([i0, i1])
            colors.append(BONE_COLORS[bi])
    ls = o3d.geometry.LineSet()
    if pts:
        ls.points = o3d.utility.Vector3dVector(np.array(pts))
        ls.lines  = o3d.utility.Vector2iVector(np.array(lines))
        ls.colors = o3d.utility.Vector3dVector(np.array(colors))
    all_pts = np.array([joints[j] for j in JOINT_NAMES if j in joints])
    joint_pcd = o3d.geometry.PointCloud()
    joint_pcd.points = o3d.utility.Vector3dVector(all_pts)
    joint_pcd.paint_uniform_color([1.0, 1.0, 1.0])
    return ls, joint_pcd


def update_skeleton_geometry(ls, joint_pcd, body, fi):
    joints = {j: body[j][fi] for j in JOINT_NAMES if j in body}
    pts, lines, colors = [], [], []
    for bi, (p, c) in enumerate(BONES):
        if p in joints and c in joints:
            i0 = len(pts); pts.append(joints[p])
            i1 = len(pts); pts.append(joints[c])
            lines.append([i0, i1])
            colors.append(BONE_COLORS[bi])
    ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines))
    ls.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    all_pts = np.array([joints[j] for j in JOINT_NAMES if j in joints], dtype=np.float64)
    joint_pcd.points = o3d.utility.Vector3dVector(all_pts)
    joint_pcd.paint_uniform_color([1.0, 1.0, 1.0])


def make_trajectory(body):
    hips = body["Hips"]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(hips.astype(np.float64))
    idx = np.stack([np.arange(len(hips)-1), np.arange(1, len(hips))], axis=1)
    ls.lines  = o3d.utility.Vector2iVector(idx)
    ls.paint_uniform_color([0.25, 0.25, 0.25])
    return ls


# ── Gaze heatmap ─────────────────────────────────────────────────────────────
#
# For each mesh point, accumulate Gaussian weights from nearby gaze hit points
# (Gaussian KDE with σ = HEATMAP_SIGMA).  The resulting density is normalised
# to [0, 1] and mapped through a 5-stop jet-like colormap:
#
#   0.00 → base mesh colour  (dim, ungazed)
#   0.25 → blue
#   0.50 → cyan
#   0.75 → green / yellow
#   1.00 → red   (peak density)

_HEATMAP_STOPS_X   = np.array([0.0,  0.25, 0.5,  0.75, 1.0 ])
_HEATMAP_STOPS_RGB = np.array([
    [0.0,  0.0,  0.5 ],   # dark blue
    [0.0,  0.3,  1.0 ],   # blue
    [0.0,  1.0,  0.5 ],   # cyan-green
    [1.0,  0.8,  0.0 ],   # yellow
    [1.0,  0.1,  0.0 ],   # red
])


def precompute_gaze_heatmap(mesh_xyz: np.ndarray,
                             mesh_rgb: np.ndarray,
                             gaze_pts_dji: np.ndarray,
                             gaze_valid: np.ndarray,
                             frame_indices: np.ndarray):
    """
    Precompute per-playback-frame Gaussian KDE contributions for an animated
    gaze heatmap.

    Parameters
    ----------
    mesh_xyz     : (N, 3) float64  mesh point positions
    mesh_rgb     : (N, 3) float64  original mesh colours  [0, 1]
    gaze_pts_dji : (T, 3) float64  gaze hit points (full data array, NaN if invalid)
    gaze_valid   : (T,) bool       True when the frame has a valid mesh-intersection
    frame_indices: (F,) int        mapping from playback frame → data row index

    Returns
    -------
    contribs : list of (idx_arr, w_arr) or None, length F
               For each playback frame, the mesh-point indices and Gaussian
               weights contributed by that frame's gaze hit (None if no hit).
    d_max    : float  global density maximum for consistent normalisation
    base     : (N, 3) float64  dimmed base mesh colours (used as the zero level)
    """
    base = mesh_rgb * 0.6

    n_hits = int(gaze_valid[frame_indices].sum())
    print(f"  Precomputing animated gaze heatmap  "
          f"({n_hits} hits, σ={HEATMAP_SIGMA}m, r={HEATMAP_RADIUS}m) …",
          end=" ", flush=True)

    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(mesh_xyz)
    tree = o3d.geometry.KDTreeFlann(pcd_tmp)
    two_sig2 = 2.0 * HEATMAP_SIGMA ** 2

    n_frames = len(frame_indices)
    contribs = [None] * n_frames
    density_final = np.zeros(len(mesh_xyz), dtype=np.float64)

    for pf, fi in enumerate(frame_indices):
        if gaze_valid[fi]:
            hit = gaze_pts_dji[fi]
            k, idx, dist2 = tree.search_radius_vector_3d(hit, HEATMAP_RADIUS)
            if k > 0:
                idx_arr = np.asarray(idx).copy()
                w_arr   = np.exp(-np.asarray(dist2) / two_sig2)
                contribs[pf] = (idx_arr, w_arr)
                density_final[idx_arr] += w_arr

    d_max = density_final.max()
    n_hot = int((density_final > 0).sum())
    print(f"{n_hot:,} mesh points affected", flush=True)
    return contribs, d_max, base


def _density_to_colors(density: np.ndarray,
                        d_max: float,
                        base: np.ndarray) -> np.ndarray:
    """
    Map an accumulated density array to blended heatmap colours.

    Parameters
    ----------
    density : (N,) float64  accumulated Gaussian KDE weights
    d_max   : float         global maximum (from precompute_gaze_heatmap)
    base    : (N, 3) float64 dimmed base mesh colours

    Returns
    -------
    colors : (N, 3) float64
    """
    if d_max == 0:
        return base
    t = np.minimum(density / d_max, 1.0)   # (N,) in [0, 1]
    heat = np.column_stack([
        np.interp(t, _HEATMAP_STOPS_X, _HEATMAP_STOPS_RGB[:, c])
        for c in range(3)
    ])
    alpha = t[:, None]
    return np.clip((1.0 - alpha) * base + alpha * heat, 0.0, 1.0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    # ── Load data ─────────────────────────────────────────────────────────────
    mesh_xyz, mesh_rgb = load_ply_sample(PLY_PATH, N_MESH_PTS)
    body, n_total = load_frames(CSV_PATH)
    frame_indices = np.arange(0, n_total, FRAME_STRIDE)
    n_frames = len(frame_indices)
    print(f"  {n_frames} playback frames  "
          f"({FRAME_STRIDE}× stride → {n_frames/TARGET_FPS:.0f}s @ {TARGET_FPS}fps)",
          flush=True)

    aria_ok      = "cam_pos" in body
    cam_pos_dji  = body.get("cam_pos")
    cam_rot_dji  = body.get("cam_rot")
    gaze_pts_dji = body.get("gaze_pt")
    gaze_valid   = body.get("gaze_valid")

    # ── Gaze heatmap precomputation ───────────────────────────────────────────
    if aria_ok:
        gaze_contribs, heatmap_d_max, heatmap_base = precompute_gaze_heatmap(
            mesh_xyz, mesh_rgb, gaze_pts_dji, gaze_valid, frame_indices)
        mesh_colors = heatmap_base   # start empty — accumulates during playback
    else:
        mesh_colors = mesh_rgb * 0.6

    # ── Build Open3D geometries ───────────────────────────────────────────────
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(mesh_xyz)
    mesh_pcd.colors = o3d.utility.Vector3dVector(mesh_colors)

    traj_ls = make_trajectory(body)
    fi0 = frame_indices[0]
    skel_ls, joint_pcd = make_skeleton_geometry(body, fi0)

    if aria_ok:
        frustum_ls   = make_frustum_ls()
        aria_traj_ls = make_aria_traj_ls(cam_pos_dji)
        gaze_ls      = make_gaze_ls()
        update_frustum_ls(frustum_ls, cam_pos_dji[fi0], cam_rot_dji[fi0])
        update_gaze_ls(gaze_ls, cam_pos_dji[fi0], gaze_pts_dji[fi0], gaze_valid[fi0])

    # ── Viewer state ──────────────────────────────────────────────────────────
    state = {
        "frame":      0,
        "paused":     False,
        "speed":      1.0,
        "last_t":     time.time(),
        "show_aria":  aria_ok,   # frustum + camera traj
        "show_gaze":  aria_ok,   # gaze ray
        "follow_cam": False,     # camera tracks skeleton hips each frame
    }

    # ── Visualizer ────────────────────────────────────────────────────────────
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=(
            "Walk Viewer  "
            "[Space=pause  ←/→=step  R=restart  [/]=speed  "
            "F=frustum  G=gaze  C=follow-cam  V=capture-frames  Q=quit]"
        ),
        width=1600, height=900,
    )

    vis.add_geometry(mesh_pcd)
    vis.add_geometry(traj_ls)
    vis.add_geometry(skel_ls)
    vis.add_geometry(joint_pcd)
    if aria_ok:
        vis.add_geometry(aria_traj_ls)
        vis.add_geometry(frustum_ls)
        vis.add_geometry(gaze_ls)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.04, 0.04, 0.06])
    opt.point_size = 1.5
    opt.line_width = 3.0
    opt.show_coordinate_frame = False

    hips_mean = body["Hips"].mean(axis=0)
    ctrl = vis.get_view_control()
    ctrl.set_lookat(hips_mean)
    ctrl.set_up([0, 0, 1])
    ctrl.set_front([0.4, -0.8, 0.5])
    ctrl.set_zoom(0.35)

    # ── Video recording state ─────────────────────────────────────────────────
    # rec_state holds mutable recording context so closures can read/write it.
    rec_state = {
        "active":       False,
        "tmp_dir":      None,    # tempfile directory for PNG frames
        "count":        0,       # number of frames captured so far
        "start_frame":  None,    # playback frame index when recording began
    }

    def _start_recording():
        """Rewind, enable follow-cam, create a timestamped frames folder, begin capture."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(FRAMES_DIR, f"rec_{timestamp}")
        os.makedirs(out_dir, exist_ok=True)
        rec_state["tmp_dir"]     = out_dir
        rec_state["count"]       = 0
        rec_state["start_frame"] = state["frame"]
        rec_state["active"]      = True
        state["follow_cam"]      = True
        state["frame"]           = 0
        state["paused"]          = False
        print(f"  Recording started → {out_dir}", flush=True)

    def _stop_recording():
        """Stop capture and report where the frames were saved."""
        rec_state["active"] = False
        n   = rec_state["count"]
        out = os.path.abspath(rec_state["tmp_dir"])
        if n == 0:
            print("  Recording stopped — no frames captured.", flush=True)
        else:
            print(f"  Recording stopped — {n} frames saved to {out}", flush=True)

    # ── Key callbacks ─────────────────────────────────────────────────────────
    GLFW_KEY_SPACE    = 32
    GLFW_KEY_LEFT     = 263
    GLFW_KEY_RIGHT    = 262
    GLFW_KEY_R        = 82
    GLFW_KEY_LBRACKET = 91
    GLFW_KEY_RBRACKET = 93
    GLFW_KEY_C        = 67
    GLFW_KEY_F        = 70
    GLFW_KEY_G        = 71
    GLFW_KEY_V        = 86
    GLFW_KEY_Q        = 81

    BG = [0.04, 0.04, 0.06]  # background colour used to "hide" geometry

    def _update_aria(fi):
        update_frustum_ls(frustum_ls, cam_pos_dji[fi], cam_rot_dji[fi])
        vis.update_geometry(frustum_ls)

    def _update_gaze(fi):
        update_gaze_ls(gaze_ls, cam_pos_dji[fi], gaze_pts_dji[fi], gaze_valid[fi])
        vis.update_geometry(gaze_ls)

    # ── Animated heatmap state ────────────────────────────────────────────────
    if aria_ok:
        _hmap = {"density": np.zeros(len(mesh_xyz), dtype=np.float64), "last_pf": -1}

        def update_heatmap(target_pf: int):
            """
            Accumulate gaze KDE contributions up to playback frame *target_pf*
            and refresh the mesh point-cloud colours.

            Forward step: only the new frames are added (O(new frames)).
            Backward step (e.g. restart / scrubbing left): full recomputation
            from frame 0 to target_pf.
            """
            reset = False
            changed = False

            if target_pf < _hmap["last_pf"]:
                # Went backwards — reset and recompute from scratch
                _hmap["density"][:] = 0.0
                reset = True
                for pf in range(target_pf + 1):
                    c = gaze_contribs[pf]
                    if c is not None:
                        _hmap["density"][c[0]] += c[1]
                        changed = True
            else:
                # Moving forward — incrementally add new frames
                for pf in range(_hmap["last_pf"] + 1, target_pf + 1):
                    c = gaze_contribs[pf]
                    if c is not None:
                        _hmap["density"][c[0]] += c[1]
                        changed = True

            _hmap["last_pf"] = target_pf

            if changed or reset:
                colors = _density_to_colors(_hmap["density"], heatmap_d_max, heatmap_base)
                mesh_pcd.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(mesh_pcd)

    def _refresh(vis):
        fi = frame_indices[state["frame"]]
        update_skeleton_geometry(skel_ls, joint_pcd, body, fi)
        vis.update_geometry(skel_ls)
        vis.update_geometry(joint_pcd)
        if aria_ok:
            update_heatmap(state["frame"])
            if state["show_aria"]:
                _update_aria(fi)
            if state["show_gaze"]:
                _update_gaze(fi)
        t_s  = body["time"][fi]
        foot = body["foot"][fi]
        extras = ""
        if aria_ok:
            p = cam_pos_dji[fi]
            extras += f"  cam=[{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}]"
            if gaze_valid[fi]:
                g = gaze_pts_dji[fi]
                extras += f"  gaze=[{g[0]:.2f} {g[1]:.2f} {g[2]:.2f}]"
        print(f"  frame {state['frame']:4d}/{n_frames}  t={t_s:.2f}s  {foot}{extras}",
              flush=True)

    def cb_space(vis):
        state["paused"] = not state["paused"]
        print("Paused" if state["paused"] else "Playing", flush=True)
        return False

    def cb_left(vis):
        state["paused"] = True
        state["frame"] = (state["frame"] - 1) % n_frames
        _refresh(vis)
        return False

    def cb_right(vis):
        state["paused"] = True
        state["frame"] = (state["frame"] + 1) % n_frames
        _refresh(vis)
        return False

    def cb_restart(vis):
        state["frame"] = 0
        state["paused"] = False
        if aria_ok:
            update_heatmap(0)
        return False

    def cb_slower(vis):
        state["speed"] = max(0.25, state["speed"] / 2)
        print(f"Speed: {state['speed']:.2f}×", flush=True)
        return False

    def cb_faster(vis):
        state["speed"] = min(4.0, state["speed"] * 2)
        print(f"Speed: {state['speed']:.2f}×", flush=True)
        return False

    def cb_frustum(vis):
        if not aria_ok: return False
        state["show_aria"] = not state["show_aria"]
        frustum_ls.paint_uniform_color(FRUSTUM_COLOR if state["show_aria"] else BG)
        aria_traj_ls.paint_uniform_color(ARIA_TRAJ_COLOR if state["show_aria"] else BG)
        vis.update_geometry(frustum_ls)
        vis.update_geometry(aria_traj_ls)
        print(f"Aria frustum: {'ON' if state['show_aria'] else 'OFF'}", flush=True)
        return False

    def cb_gaze(vis):
        if not aria_ok: return False
        state["show_gaze"] = not state["show_gaze"]
        if not state["show_gaze"]:
            # Collapse gaze ray to invisible (degenerate)
            fi = frame_indices[state["frame"]]
            cp = cam_pos_dji[fi]
            gaze_ls.points = o3d.utility.Vector3dVector(np.tile(cp, (8, 1)))
            vis.update_geometry(gaze_ls)
        else:
            fi = frame_indices[state["frame"]]
            _update_gaze(fi)
        print(f"Gaze ray: {'ON' if state['show_gaze'] else 'OFF'}", flush=True)
        return False

    def cb_follow(vis):
        state["follow_cam"] = not state["follow_cam"]
        print(f"Camera follow: {'ON' if state['follow_cam'] else 'OFF'}", flush=True)
        return False

    def cb_record(vis):
        if rec_state["active"]:
            _stop_recording()
        else:
            _start_recording()
        return False

    def cb_quit(vis):
        if rec_state["active"]:
            _stop_recording()
        vis.destroy_window()
        return False

    vis.register_key_callback(GLFW_KEY_SPACE,    cb_space)
    vis.register_key_callback(GLFW_KEY_LEFT,     cb_left)
    vis.register_key_callback(GLFW_KEY_RIGHT,    cb_right)
    vis.register_key_callback(GLFW_KEY_R,        cb_restart)
    vis.register_key_callback(GLFW_KEY_LBRACKET, cb_slower)
    vis.register_key_callback(GLFW_KEY_RBRACKET, cb_faster)
    vis.register_key_callback(GLFW_KEY_C,        cb_follow)
    vis.register_key_callback(GLFW_KEY_F,        cb_frustum)
    vis.register_key_callback(GLFW_KEY_G,        cb_gaze)
    vis.register_key_callback(GLFW_KEY_V,        cb_record)
    vis.register_key_callback(GLFW_KEY_Q,        cb_quit)

    fov_deg = 2 * np.degrees(np.arctan(_TAN_HX))
    print("\n── Walk Viewer ──────────────────────────────────────────────────────")
    print("  Space      pause / resume")
    print("  ← / →      step frame-by-frame (auto-pauses)")
    print("  R          restart")
    print("  [ / ]      half / double speed")
    print("  C          toggle camera-follow  (camera tracks skeleton hips)")
    print("  F          toggle Aria camera frustum + trajectory")
    print("  G          toggle gaze ray  (magenta = mesh intersection)")
    print(f"  V          start / stop frame capture  → {FRAMES_DIR}/<timestamp>/")
    print("  Q or Esc   quit  (stops any active recording first)")
    print("  Mouse      left-drag=rotate  right-drag=pan  scroll=zoom")
    if aria_ok:
        n_hit = int(gaze_valid.sum())
        print(f"\n  Aria frustum : ON (yellow)  "
              f"near={FRUSTUM_NEAR}m  far={FRUSTUM_FAR}m  FOV≈{fov_deg:.0f}°×{fov_deg:.0f}°")
        print(f"  Gaze ray     : ON (magenta)  "
              f"{n_hit}/{n_total} frames have mesh-intersection gaze")
    else:
        print("\n  Aria overlay: DISABLED (run align_body.py to precompute Aria columns)")
    print("─────────────────────────────────────────────────────────────────────")

    # ── Animation loop ────────────────────────────────────────────────────────
    target_dt = 1.0 / TARGET_FPS

    while True:
        if not vis.poll_events():
            break

        frame_advanced = False

        if not state["paused"]:
            now = time.time()
            dt  = now - state["last_t"]
            if dt >= target_dt / state["speed"]:
                prev_frame    = state["frame"]
                state["frame"] = (state["frame"] + 1) % n_frames
                fi = frame_indices[state["frame"]]

                update_skeleton_geometry(skel_ls, joint_pcd, body, fi)
                vis.update_geometry(skel_ls)
                vis.update_geometry(joint_pcd)

                if aria_ok:
                    update_heatmap(state["frame"])
                    if state["show_aria"]:
                        _update_aria(fi)
                    if state["show_gaze"]:
                        _update_gaze(fi)

                state["last_t"] = now
                frame_advanced  = True

                # Auto-stop recording after one full pass (loop back to 0)
                if rec_state["active"] and state["frame"] == 0 and prev_frame != 0:
                    _stop_recording()

        # ── Camera follow ─────────────────────────────────────────────────────
        # Keep lookat locked to the skeleton hips when follow-cam is active.
        # Must run every tick (not just on frame advance) so panning is smooth.
        if state["follow_cam"]:
            fi = frame_indices[state["frame"]]
            ctrl.set_lookat(body["Hips"][fi].tolist())

        vis.update_renderer()

        # ── Frame capture for video ───────────────────────────────────────────
        # Capture *after* update_renderer so the GPU output is settled.
        if rec_state["active"] and frame_advanced:
            frame_path = os.path.join(
                rec_state["tmp_dir"],
                f"frame_{rec_state['count']:06d}.png",
            )
            vis.capture_screen_image(frame_path, do_render=False)
            rec_state["count"] += 1

    # If the user closes the window while recording, save what we have.
    if rec_state["active"]:
        _stop_recording()

    print("Viewer closed.")


if __name__ == "__main__":
    main()
