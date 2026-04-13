#!/usr/bin/env python3
"""
rigid_align.py
================
Fits Shadow MoCap heel positions to DJI ground-contact heel positions using
a windowed Umeyama similarity transform with Gaussian smoothing across windows.

Three alignment modes are available:

  Fixed-window (original):
    For each step i a local Umeyama is fit on the 'window_size' nearest matched
    pairs centred on i.  The resulting per-step (R, t, c) are decomposed into
    (quaternion, translation, scale), Gaussian-smoothed with sigma=smooth_sigma
    steps, then reconstructed into smooth per-step transforms.

  Adaptive-window (default):
    A two-pass scheme driven by per-step residuals.

    Pass 1  – A broad initial window (base_window) is used to compute a
              preliminary alignment and residual at every valid step.
    Sizing  – The residual at each step is compared to a threshold
              (mean + residual_threshold_factor × std).  Steps whose residual
              exceeds the threshold receive a smaller, tighter window
              (down to min_window) so the local fit can track rapid changes
              (turns, elevation changes, etc.).  Window sizes are themselves
              Gaussian-smoothed to avoid abrupt transitions.
    Pass 2  – The full per-step alignment is re-run using the adaptive window
              sizes, then Gaussian-smoothed as in the fixed-window mode.

  Rotation-window (new):
    Segments the trajectory into contiguous windows at points of significant
    body reorientation (heading change above --rotation-threshold degrees).
    A separate rigid transform is fitted per window, then Gaussian-smoothed
    across window boundaries for a seamless result.

  Global-rotation modifier (--global-rotation):
    Can be combined with fixed-window or adaptive modes (not rotation-window).
    A single rotation R is fitted once from all valid pairs; the per-window
    alignment then only optimises for translation.  This prevents the
    orientation from drifting across windows while still allowing the
    trajectory to track local positional offsets.

In addition to the 3-D Open3D viewer, a top-down 2-D matplotlib plot is always
saved showing the trajectory coloured by rotation-detected window, DJI hits,
and window boundary markers.  The plot is saved to topdown_2d.png.

Usage:
    python3 umeyama_align.py [options]

Key options:
    --fixed-window          Use the original fixed-window mode
    --rotation-window       Use rotation-based windowing mode
    --window-size  W        Window size for fixed mode (default 30)
    --base-window  W        Max window for adaptive mode (default 60)
    --min-window   W        Min window for adaptive mode (default 8)
    --residual-threshold K  Threshold = mean + K*std (default 1.5)
    --smooth-sigma S        Gaussian σ in steps (default 5)
    --rotation-threshold D  Heading change (deg) to start new window (default 30)
    --rotation-min-steps M  Minimum steps per rotation window (default 20)
    --heading-smooth S      σ for smoothing trajectory before heading calc (default 5)
    --global-rotation       One global rotation + windowed translation (fixed/adaptive only)
    --no-2d-plot            Skip saving the top-down 2D matplotlib figure

Controls (Open3D viewer)
------------------------
  Mouse left-drag   – rotate
  Mouse right-drag  – pan
  Scroll            – zoom
  Q / Esc           – quit
"""

import argparse

import matplotlib
matplotlib.use("Agg")   # save to file without needing a display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import open3d as o3d

N_PLY_SAMPLE = 3_000_000


def _augment_with_head(w_src, w_dst, head_src_scaled, head_dst_all,
                       head_valid_idx, step_lo, step_hi):
    """Append valid Aria head correspondences in step range [step_lo, step_hi)."""
    if head_valid_idx is None or len(head_valid_idx) == 0:
        return w_src, w_dst
    in_range = head_valid_idx[(head_valid_idx >= step_lo) & (head_valid_idx < step_hi)]
    if len(in_range) == 0:
        return w_src, w_dst
    return (np.vstack([w_src, head_src_scaled[in_range]]),
            np.vstack([w_dst, head_dst_all[in_range]]))


def load_aria_head_positions(heels_df,
                              alignment_npz_path,
                              frame_positions_path,
                              dji_aria_matches_path,
                              time_aligned_path,
                              detected_steps_csv):
    """
    For each step in heels_df, return:
      - shadow Head position (src) from detected_steps.csv at the step's mid-frame
      - Aria camera position in DJI world space (dst)

    Mapping chain:
      step i → mid shadow frame
             → nearest row in time_aligned_steps.csv → dji_frame
             → aria_frame in dji_aria_frame_matches.csv
             → world_position in frame_positions.json
             → DJI world via inv(T_total from alignment_transform.npz)

    Returns
    -------
    head_src_all : (N, 3) shadow Head positions in Shadow coordinate space
    head_dst_all : (N, 3) Aria camera positions in DJI world (NaN where no match)
    head_valid   : (N,) bool
    """
    import json
    import os

    N = len(heels_df)
    head_src_all = np.full((N, 3), np.nan)
    head_dst_all = np.full((N, 3), np.nan)
    head_valid   = np.zeros(N, dtype=bool)

    # ── Check required files exist ────────────────────────────────────────────
    required = {
        "alignment_npz":    alignment_npz_path,
        "frame_positions":  frame_positions_path,
        "dji_aria_matches": dji_aria_matches_path,
        "time_aligned":     time_aligned_path,
    }
    for label, path in required.items():
        if path is None or not os.path.exists(path):
            display = path if path is not None else f"<{label} not provided>"
            print(f"  [aria] {display!r} not found — skipping Aria head constraint")
            return head_src_all, head_dst_all, head_valid

    # ── Load alignment transform ──────────────────────────────────────────────
    npz     = np.load(alignment_npz_path)
    T_total = npz['T_total']           # DJI → Aria world  (4×4)
    T_inv   = np.linalg.inv(T_total)   # Aria world → DJI world

    # ── Load time-alignment tables ────────────────────────────────────────────
    tal     = pd.read_csv(time_aligned_path)
    matches = pd.read_csv(dji_aria_matches_path)

    # ── Load shadow joint data (for head positions) ───────────────────────────
    detected = pd.read_csv(detected_steps_csv)
    det_frames   = detected['frame'].values
    det_head_xyz = detected[['Head_x', 'Head_y', 'Head_z']].values
    det_sorted   = np.argsort(det_frames)
    det_frames_s = det_frames[det_sorted]

    # ── Load Aria frame positions ─────────────────────────────────────────────
    with open(frame_positions_path) as f:
        frame_positions = json.load(f)
    fp_name_to_idx = {fp['frame_name']: i for i, fp in enumerate(frame_positions)}

    # ── Build fast lookup for time_aligned_steps ──────────────────────────────
    tal_frames     = tal['frame'].values
    tal_dji_frames = tal['dji_frame'].values
    tal_sorted     = np.argsort(tal_frames)
    tal_frames_s   = tal_frames[tal_sorted]
    tal_frame_min, tal_frame_max = int(tal_frames.min()), int(tal_frames.max())

    aria_frame_names = matches['aria_frame'].values   # index = dji_frame

    # ── Per-step mapping ──────────────────────────────────────────────────────
    for i in range(N):
        row = heels_df.iloc[i]
        mid_frame = int((int(row['frame_start']) + int(row['frame_end'])) // 2)

        # Shadow head position from detected_steps.csv
        det_pos = int(np.clip(np.searchsorted(det_frames_s, mid_frame),
                              0, len(det_frames_s) - 1))
        head_src_all[i] = det_head_xyz[det_sorted[det_pos]]

        # Aria position: only available for steps within the DJI-aligned range
        if mid_frame < tal_frame_min - 300 or mid_frame > tal_frame_max + 300:
            continue

        tal_pos  = int(np.clip(np.searchsorted(tal_frames_s, mid_frame),
                               0, len(tal_frames_s) - 1))
        tal_row  = tal_sorted[tal_pos]
        if abs(int(tal_frames[tal_row]) - mid_frame) > 300:  # > 3 s gap
            continue

        dji_frame_idx = int(tal_dji_frames[tal_row])
        if dji_frame_idx < 0 or dji_frame_idx >= len(aria_frame_names):
            continue

        aria_name = aria_frame_names[dji_frame_idx]
        if aria_name not in fp_name_to_idx:
            continue

        fp      = frame_positions[fp_name_to_idx[aria_name]]
        wp      = fp['world_position']
        aria_h  = np.array([wp['x'], wp['y'], wp['z'], 1.0])
        dji_pos = (T_inv @ aria_h)[:3]

        head_dst_all[i] = dji_pos

        # Sanity check: Aria camera must be at least MIN_ARIA_HEAD_HEIGHT above
        # the DJI ground contact point.  If the alignment transform has error,
        # the transformed Aria position can fall below the terrain surface for
        # the latter portion of the walk — those constraints are actively harmful
        # (they pull the head downward, fighting the heel-to-ground constraints).
        MIN_ARIA_HEAD_HEIGHT = 0.3   # metres above DJI heel-contact Z
        ground_z = row['median_hit_z']   # NaN if missing (pandas fills empty cells)
        if np.isnan(float(ground_z)) or (dji_pos[2] - float(ground_z)) >= MIN_ARIA_HEAD_HEIGHT:
            head_valid[i] = True

    n_valid  = int(head_valid.sum())
    n_mapped = int(np.isfinite(head_dst_all[:, 0]).sum())
    n_below  = n_mapped - n_valid
    print(f"  [aria] Head correspondences: {n_valid}/{N} steps valid "
          f"({n_below} discarded — Aria cam below/near DJI ground)")
    if n_valid > 0:
        pts = head_dst_all[head_valid]
        print(f"  [aria] DJI head pos range: "
              f"X=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}]  "
              f"Y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]  "
              f"Z=[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
    return head_src_all, head_dst_all, head_valid


# ── Umeyama (single window) ───────────────────────────────────────────────────

def umeyama(src, dst):
    """
    Least-squares similarity:  dst ≈ c · R · src + t
    src, dst : (N, 3) – must have N >= 3
    Returns (R 3×3, t 3, c scalar)
    """
    n, m = src.shape
    mu_src = src.mean(0);  mu_dst = dst.mean(0)
    src_c  = src - mu_src; dst_c  = dst - mu_dst
    var_src = np.sum(src_c ** 2) / n
    K       = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(K)
    D = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[m - 1, m - 1] = -1.0
    R = U @ D @ Vt
    c = float(np.sum(S * np.diag(D))) / var_src
    t = mu_dst - c * (R @ mu_src)
    return R, t, c


def apply_transform(pts, R, t, c):
    return (c * (R @ pts.T)).T + t

def windowed_rigid(src_all, dst_all, valid_mask, window_size, smooth_sigma,
                   fixed_scale=0.9,
                   head_src_all=None, head_dst_all=None, head_valid=None):
    """
    Per-step windowed rigid fit with Gaussian-smoothed stitching.

    The source points are first scaled by `fixed_scale`, then a rigid (rotation + translation)
    transform is found to align them to the destination points. No per-window scaling is applied.

    Optional Aria head correspondences (head_src_all, head_dst_all, head_valid) are
    appended to each local window to anchor the head position against the Aria camera.

    Parameters
    ----------
    src_all    : (N, 3) shadow heel positions for all N steps
    dst_all    : (N, 3) DJI heel positions (NaN where missing)
    valid_mask : (N,) bool – True where dst is valid
    window_size : int – number of matched pairs used per local fit
    smooth_sigma : float – Gaussian σ in steps for smoothing transforms
    fixed_scale : float – pre-scaling applied to src_all before alignment
    head_src_all : (N, 3) or None – shadow Head positions
    head_dst_all : (N, 3) or None – Aria camera positions in DJI world
    head_valid   : (N,) bool or None

    Returns
    -------
    aligned : (N, 3) smoothly-aligned shadow positions
    quats   : (N, 4) smoothed quaternions (for diagnostics)
    trans   : (N, 3) smoothed translations
    """
    N = len(src_all)
    valid_idx = np.where(valid_mask)[0]   # indices of matched pairs
    src_v = src_all[valid_idx] * fixed_scale  # scale first
    dst_v = dst_all[valid_idx]

    half_w = window_size // 2
    n_valid = len(valid_idx)

    head_valid_idx  = np.where(head_valid)[0] if head_valid is not None else None
    head_src_scaled = head_src_all * fixed_scale if head_src_all is not None else None

    raw_quats = np.zeros((N, 4))
    raw_trans = np.zeros((N, 3))

    # ── Per-step local rigid fit ─────────────────────────────────────────────
    for i in range(N):
        pos = np.searchsorted(valid_idx, i)
        pos = int(np.clip(pos, 0, n_valid - 1))

        lo = max(0, pos - half_w)
        hi = min(n_valid, lo + window_size)
        lo = max(0, hi - window_size)

        w_src = src_v[lo:hi]
        w_dst = dst_v[lo:hi]

        # Augment with Aria head pairs in the same step range
        if head_valid_idx is not None and lo < n_valid:
            step_lo = int(valid_idx[lo])
            step_hi = int(valid_idx[min(hi, n_valid) - 1]) + 1
            w_src, w_dst = _augment_with_head(
                w_src, w_dst, head_src_scaled, head_dst_all,
                head_valid_idx, step_lo, step_hi)

        if len(w_src) < 3:
            # Degenerate window – use global rigid fit as fallback
            fb_src, fb_dst = _augment_with_head(
                src_v, dst_v, head_src_scaled, head_dst_all, head_valid_idx, 0, N)
            R, t = umeyama_rigid(fb_src, fb_dst)
        else:
            R, t = umeyama_rigid(w_src, w_dst)

        raw_quats[i] = Rotation.from_matrix(R).as_quat()
        raw_trans[i] = t

    # ── Gaussian smoothing ─────────────────────────────────────────────────
    if smooth_sigma > 0:
        sm_trans = gaussian_filter1d(raw_trans, smooth_sigma, axis=0)
        sm_quats = gaussian_filter1d(raw_quats, smooth_sigma, axis=0)
        sm_quats /= np.linalg.norm(sm_quats, axis=1, keepdims=True)
    else:
        sm_trans = raw_trans
        sm_quats = raw_quats

    # ── Apply per-step transforms ───────────────────────────────────────────
    aligned = np.zeros((N, 3))
    for i in range(N):
        R_i = Rotation.from_quat(sm_quats[i]).as_matrix()
        aligned[i] = apply_transform_pre_scaled(src_all[[i]] * fixed_scale, R_i, sm_trans[i])[0]

    return aligned, sm_quats, sm_trans, np.ones(len(sm_quats))*.9


def windowed_rigid_adaptive(
    src_all, dst_all, valid_mask,
    base_window=60,
    min_window=8,
    residual_threshold_factor=1.5,
    smooth_sigma=5.0,
    fixed_scale=0.9,
    head_src_all=None, head_dst_all=None, head_valid=None,
):
    """
    Two-pass adaptive windowed rigid alignment driven by per-step residuals.

    In regions where the alignment struggles (turns, height changes, etc.) the
    residuals will be elevated.  This function automatically tightens the local
    window there so the fit can track rapid motion, while keeping a wide window
    in easy flat/straight regions for stability.

    Optional Aria head correspondences (head_src_all, head_dst_all, head_valid) are
    appended to each local window to anchor the head position against the Aria camera.

    Parameters
    ----------
    src_all                  : (N, 3) Shadow heel positions
    dst_all                  : (N, 3) DJI heel positions (NaN where missing)
    valid_mask               : (N,) bool
    base_window              : int   – largest window used (low-residual regions)
    min_window               : int   – smallest window used (high-residual regions)
    residual_threshold_factor: float – threshold = mean + k * std of pass-1 residuals
    smooth_sigma             : float – Gaussian σ for smoothing transforms
    fixed_scale              : float – pre-scaling applied to src before alignment
    head_src_all : (N, 3) or None – shadow Head positions
    head_dst_all : (N, 3) or None – Aria camera positions in DJI world
    head_valid   : (N,) bool or None

    Returns
    -------
    aligned         : (N, 3) aligned Shadow positions
    sm_quats        : (N, 4) smoothed quaternions
    sm_trans        : (N, 3) smoothed translations
    adaptive_windows: (N,)   int, window size used per step (for diagnostics)
    residuals_pass1 : (N,)   float, per-step residual from pass 1
    """
    N = len(src_all)
    valid_idx = np.where(valid_mask)[0]
    src_v = src_all[valid_idx] * fixed_scale
    dst_v = dst_all[valid_idx]
    n_valid = len(valid_idx)

    head_valid_idx  = np.where(head_valid)[0] if head_valid is not None else None
    head_src_scaled = head_src_all * fixed_scale if head_src_all is not None else None

    # ── Pass 1: broad window → per-valid-point residuals ─────────────────────
    print(f"  [adaptive] Pass 1: base_window={base_window} …", flush=True)
    residuals_v = np.zeros(n_valid)

    for vi in range(n_valid):
        half_w = base_window // 2
        lo = max(0, vi - half_w)
        hi = min(n_valid, lo + base_window)
        lo = max(0, hi - base_window)

        w_src = src_v[lo:hi]
        w_dst = dst_v[lo:hi]

        # Augment with head pairs in this step range
        if head_valid_idx is not None and lo < n_valid:
            step_lo = int(valid_idx[lo])
            step_hi = int(valid_idx[min(hi, n_valid) - 1]) + 1
            w_src, w_dst = _augment_with_head(
                w_src, w_dst, head_src_scaled, head_dst_all,
                head_valid_idx, step_lo, step_hi)

        if len(w_src) < 3:
            fb_src, fb_dst = _augment_with_head(
                src_v, dst_v, head_src_scaled, head_dst_all, head_valid_idx, 0, N)
            R, t = umeyama_rigid(fb_src, fb_dst)
        else:
            R, t = umeyama_rigid(w_src, w_dst)

        pred = apply_transform_pre_scaled(src_v[[vi]], R, t)[0]
        residuals_v[vi] = np.linalg.norm(pred - dst_v[vi])

    # Interpolate residuals to all N steps (linear, extrapolate with edge values)
    residuals_all = np.interp(np.arange(N), valid_idx, residuals_v)

    # ── Compute per-step adaptive window sizes ────────────────────────────────
    r_mean = residuals_all.mean()
    r_std  = residuals_all.std()
    threshold = r_mean + residual_threshold_factor * r_std

    print(f"  [adaptive] Residual pass-1:  mean={r_mean:.4f}  std={r_std:.4f}  "
          f"threshold={threshold:.4f}", flush=True)

    # t_val: 0 → low residual (use base_window), 1 → high residual (use min_window)
    t_val = np.clip((residuals_all - r_mean) / (threshold - r_mean + 1e-9), 0.0, 1.0)
    adaptive_windows_raw = np.round(
        base_window - t_val * (base_window - min_window)
    ).astype(int)
    adaptive_windows_raw = np.clip(adaptive_windows_raw, min_window, base_window)

    # Smooth window sizes so they transition gradually
    if smooth_sigma > 0:
        adaptive_windows = np.round(
            gaussian_filter1d(adaptive_windows_raw.astype(float), smooth_sigma)
        ).astype(int)
        adaptive_windows = np.clip(adaptive_windows, min_window, base_window)
    else:
        adaptive_windows = adaptive_windows_raw

    n_tight = int((adaptive_windows < base_window * 0.8).sum())
    print(f"  [adaptive] Tight windows (<{int(base_window*0.8)} steps): "
          f"{n_tight}/{N} steps  "
          f"range=[{adaptive_windows.min()}, {adaptive_windows.max()}]", flush=True)

    # ── Pass 2: re-fit with adaptive window sizes ─────────────────────────────
    print(f"  [adaptive] Pass 2: per-step adaptive fit …", flush=True)
    raw_quats = np.zeros((N, 4))
    raw_trans = np.zeros((N, 3))

    for i in range(N):
        pos = int(np.clip(np.searchsorted(valid_idx, i), 0, n_valid - 1))
        w = int(adaptive_windows[i])
        half_w = w // 2

        lo = max(0, pos - half_w)
        hi = min(n_valid, lo + w)
        lo = max(0, hi - w)

        w_src = src_v[lo:hi]
        w_dst = dst_v[lo:hi]

        # Augment with Aria head pairs in the same step range
        if head_valid_idx is not None and lo < n_valid:
            step_lo = int(valid_idx[lo])
            step_hi = int(valid_idx[min(hi, n_valid) - 1]) + 1
            w_src, w_dst = _augment_with_head(
                w_src, w_dst, head_src_scaled, head_dst_all,
                head_valid_idx, step_lo, step_hi)

        if len(w_src) < 3:
            fb_src, fb_dst = _augment_with_head(
                src_v, dst_v, head_src_scaled, head_dst_all, head_valid_idx, 0, N)
            R, t = umeyama_rigid(fb_src, fb_dst)
        else:
            R, t = umeyama_rigid(w_src, w_dst)

        raw_quats[i] = Rotation.from_matrix(R).as_quat()
        raw_trans[i] = t

    # ── Gaussian smoothing ─────────────────────────────────────────────────────
    if smooth_sigma > 0:
        sm_trans = gaussian_filter1d(raw_trans, smooth_sigma, axis=0)
        sm_quats = gaussian_filter1d(raw_quats, smooth_sigma, axis=0)
        sm_quats /= np.linalg.norm(sm_quats, axis=1, keepdims=True)
    else:
        sm_trans = raw_trans
        sm_quats = raw_quats

    # ── Apply per-step transforms ──────────────────────────────────────────────
    aligned = np.zeros((N, 3))
    for i in range(N):
        R_i = Rotation.from_quat(sm_quats[i]).as_matrix()
        aligned[i] = apply_transform_pre_scaled(
            src_all[[i]] * fixed_scale, R_i, sm_trans[i]
        )[0]

    return aligned, sm_quats, sm_trans, adaptive_windows, residuals_all


def apply_transform_pre_scaled(pts, R, t):
    """
    Apply rotation + translation to pre-scaled points.
    """
    return pts @ R.T + t


def umeyama_rigid(src, dst):
    """
    Compute rigid (rotation + translation) alignment from src → dst using SVD.
    src, dst: (N, 3)
    Returns R, t
    """
    assert src.shape == dst.shape
    N = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst
    S = Y.T @ X / N
    U, _, Vt = np.linalg.svd(S)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = mu_dst - mu_src @ R.T
    return R, t


# ── Rotation-window detection ────────────────────────────────────────────────

def detect_rotation_windows(
    positions,
    threshold_deg=30.0,
    min_window_steps=20,
    heading_smooth_sigma=5.0,
):
    """
    Partition a trajectory into contiguous windows separated by significant
    body reorientation (heading change above threshold_deg).

    Heading is computed in the horizontal X-Y plane.  Rather than comparing
    adjacent-frame directions (which spreads a sharp turn over many steps),
    we compare the *before* heading (direction from i-L to i) against the
    *after* heading (direction from i to i+L) at every step.  This gives a
    strong, localised signal at the actual turn location.

    Parameters
    ----------
    positions            : (N, 3) trajectory — X and Y used for heading
    threshold_deg        : float – before-vs-after heading change (degrees)
                           required to open a new window (default 30)
    min_window_steps     : int   – minimum steps between consecutive breakpoints
    heading_smooth_sigma : float – Gaussian σ applied to X/Y before heading
                           computation; also used to smooth the angular-delta
                           signal before peak-finding

    Returns
    -------
    window_labels : (N,) int   – zero-based window index per step
    breakpoints   : list[int]  – step indices where new windows begin (0 first)
    heading_deg   : (N,) float – local heading (degrees) at each step
                                 (forward-direction, after smoothing)
    angular_delta : (N,) float – before-vs-after heading change magnitude (°)
    """
    N = len(positions)

    # ── Smooth horizontal positions ───────────────────────────────────────────
    sigma = max(heading_smooth_sigma, 0.0)
    if sigma > 0:
        sx = gaussian_filter1d(positions[:, 0].astype(float), sigma)
        sy = gaussian_filter1d(positions[:, 1].astype(float), sigma)
    else:
        sx = positions[:, 0].astype(float).copy()
        sy = positions[:, 1].astype(float).copy()

    # ── Look-back / look-ahead half-span ─────────────────────────────────────
    # L controls how far before and after each step we look to estimate heading.
    # Larger L → more stable but slower to detect the exact turn location.
    L = max(8, int(heading_smooth_sigma * 2))

    angular_delta = np.zeros(N)
    heading_deg   = np.zeros(N)   # forward-looking heading (for display)

    for i in range(N):
        # --- Forward heading (used for display only) ---
        j_fwd = min(i + max(5, int(sigma)), N - 1)
        dx_f = sx[j_fwd] - sx[i];  dy_f = sy[j_fwd] - sy[i]
        if np.hypot(dx_f, dy_f) > 1e-8:
            heading_deg[i] = np.degrees(np.arctan2(dy_f, dx_f))
        else:
            heading_deg[i] = heading_deg[i - 1] if i > 0 else 0.0

        # --- Before heading: direction from (i-L) to i ---
        i0 = max(0, i - L)
        dx_b = sx[i] - sx[i0];  dy_b = sy[i] - sy[i0]
        dist_b = np.hypot(dx_b, dy_b)
        h_before = np.degrees(np.arctan2(dy_b, dx_b)) if dist_b > 1e-8 else None

        # --- After heading: direction from i to (i+L) ---
        i1 = min(N - 1, i + L)
        dx_a = sx[i1] - sx[i];  dy_a = sy[i1] - sy[i]
        dist_a = np.hypot(dx_a, dy_a)
        h_after = np.degrees(np.arctan2(dy_a, dx_a)) if dist_a > 1e-8 else None

        if h_before is not None and h_after is not None:
            diff = h_after - h_before
            angular_delta[i] = abs(((diff + 180.0) % 360.0) - 180.0)
        # else: stays 0 near trajectory edges

    # ── Smooth angular delta to suppress noise spikes ────────────────────────
    if sigma > 0:
        smooth_delta = gaussian_filter1d(angular_delta, sigma)
    else:
        smooth_delta = angular_delta.copy()

    # ── Find breakpoints at local maxima above threshold ─────────────────────
    # We walk forward and pick the first step in any run that exceeds the
    # threshold, then enforce the minimum window gap.
    breakpoints = [0]
    last_break  = 0
    in_event    = False
    peak_i      = -1
    peak_v      = 0.0

    for i in range(1, N):
        if smooth_delta[i] >= threshold_deg:
            if not in_event:
                in_event = True
                peak_i   = i
                peak_v   = smooth_delta[i]
            elif smooth_delta[i] > peak_v:
                peak_i = i
                peak_v = smooth_delta[i]
        else:
            if in_event:
                # Event ended — register the peak if the gap is large enough
                if (peak_i - last_break) >= min_window_steps:
                    breakpoints.append(peak_i)
                    last_break = peak_i
                in_event = False
                peak_i   = -1
                peak_v   = 0.0
    # Handle event still open at the end of the sequence
    if in_event and (peak_i - last_break) >= min_window_steps:
        breakpoints.append(peak_i)

    # ── Build window labels ───────────────────────────────────────────────────
    window_labels = np.zeros(N, dtype=int)
    for w_idx in range(len(breakpoints)):
        start = breakpoints[w_idx]
        end   = breakpoints[w_idx + 1] if w_idx + 1 < len(breakpoints) else N
        window_labels[start:end] = w_idx

    n_windows = int(window_labels.max()) + 1
    print(f"  [rotation-detect] {n_windows} windows  "
          f"threshold={threshold_deg}°  min_steps={min_window_steps}  "
          f"L={L}  heading_smooth_sigma={sigma}")
    for w in range(n_windows):
        wm  = window_labels == w
        idx = np.where(wm)[0]
        print(f"    window {w:3d}: steps {idx[0]:4d}–{idx[-1]:4d}  "
              f"({wm.sum():4d} steps)")

    return window_labels, breakpoints, heading_deg, angular_delta


# ── Rotation-window alignment ─────────────────────────────────────────────────

def windowed_by_rotation(
    src_all,
    dst_all,
    valid_mask,
    window_labels,
    smooth_sigma=5.0,
    fixed_scale=0.9,
    head_src_all=None, head_dst_all=None, head_valid=None,
):
    """
    Per-window rigid alignment driven by rotation-detected window segments.

    Each window gets its own rigid (R, t) fit from all valid matched pairs
    within that window.  The per-step (quaternion, translation) arrays are then
    Gaussian-smoothed across window boundaries so the result is seamless.

    Optional Aria head correspondences are included in each window's fit.

    Parameters
    ----------
    src_all       : (N, 3) Shadow heel positions
    dst_all       : (N, 3) DJI heel positions (NaN where missing)
    valid_mask    : (N,) bool
    window_labels : (N,) int – output of detect_rotation_windows
    smooth_sigma  : float – Gaussian σ for smoothing across window boundaries
    fixed_scale   : float – pre-scaling applied to src before alignment
    head_src_all  : (N, 3) or None – shadow Head positions
    head_dst_all  : (N, 3) or None – Aria camera positions in DJI world
    head_valid    : (N,) bool or None

    Returns
    -------
    aligned      : (N, 3) aligned Shadow positions
    sm_quats     : (N, 4) smoothed quaternions
    sm_trans     : (N, 3) smoothed translations
    window_labels: (N,) int – passed through for convenience
    """
    N = len(src_all)
    src_scaled = src_all * fixed_scale
    n_windows  = int(window_labels.max()) + 1

    head_valid_idx  = np.where(head_valid)[0] if head_valid is not None else None
    head_src_scaled = head_src_all * fixed_scale if head_src_all is not None else None

    raw_quats = np.zeros((N, 4))
    raw_trans = np.zeros((N, 3))

    # Global fallback (augmented with head pairs)
    fb_src, fb_dst = _augment_with_head(
        src_scaled[valid_mask], dst_all[valid_mask],
        head_src_scaled, head_dst_all, head_valid_idx, 0, N)
    R_global, t_global = umeyama_rigid(fb_src, fb_dst)

    for w in range(n_windows):
        w_step_mask = window_labels == w
        w_valid     = w_step_mask & valid_mask

        # Collect heel + head pairs for this rotation window
        w_src = src_scaled[w_valid]
        w_dst = dst_all[w_valid]
        if head_valid_idx is not None:
            h_in_w = head_valid_idx[w_step_mask[head_valid_idx]]
            if len(h_in_w):
                w_src = np.vstack([w_src, head_src_scaled[h_in_w]])
                w_dst = np.vstack([w_dst, head_dst_all[h_in_w]])

        if len(w_src) >= 3:
            R_w, t_w = umeyama_rigid(w_src, w_dst)
        else:
            print(f"  [rotation-window] window {w}: only {len(w_src)} pairs "
                  f"— using global fallback")
            R_w, t_w = R_global, t_global

        q_w = Rotation.from_matrix(R_w).as_quat()
        for i in np.where(w_step_mask)[0]:
            raw_quats[i] = q_w
            raw_trans[i] = t_w

    # ── Gaussian smoothing across window boundaries ───────────────────────────
    if smooth_sigma > 0:
        sm_trans = gaussian_filter1d(raw_trans, smooth_sigma, axis=0)
        sm_quats = gaussian_filter1d(raw_quats, smooth_sigma, axis=0)
        norms = np.linalg.norm(sm_quats, axis=1, keepdims=True)
        sm_quats /= np.where(norms > 0, norms, 1.0)
    else:
        sm_trans = raw_trans
        sm_quats = raw_quats

    # ── Apply per-step transforms ─────────────────────────────────────────────
    aligned = np.zeros((N, 3))
    for i in range(N):
        R_i = Rotation.from_quat(sm_quats[i]).as_matrix()
        aligned[i] = apply_transform_pre_scaled(
            src_all[[i]] * fixed_scale, R_i, sm_trans[i]
        )[0]

    return aligned, sm_quats, sm_trans, window_labels


# ── Global-rotation windowed alignment ───────────────────────────────────────

def windowed_translation_global_rot(
    src_all, dst_all, valid_mask, window_size, smooth_sigma, fixed_scale=0.9,
    head_src_all=None, head_dst_all=None, head_valid=None,
):
    """
    Fixed-window per-step translation with one global rotation.

    1. Fit a single global rotation R from all valid (src, dst) pairs
       (augmented with Aria head pairs if provided).
    2. Pre-rotate all source points by R.
    3. For each step, compute the local-window mean translation (dst − R·src),
       including head pairs in the window.
    4. Gaussian-smooth translations; apply per step.

    Parameters
    ----------
    src_all     : (N, 3) Shadow heel positions
    dst_all     : (N, 3) DJI heel positions (NaN where missing)
    valid_mask  : (N,) bool
    window_size : int – matched pairs used per local translation fit
    smooth_sigma: float – Gaussian σ in steps
    fixed_scale : float – pre-scaling applied to src before alignment
    head_src_all : (N, 3) or None – shadow Head positions
    head_dst_all : (N, 3) or None – Aria camera positions in DJI world
    head_valid   : (N,) bool or None

    Returns
    -------
    aligned   : (N, 3) aligned Shadow positions
    sm_quats  : (N, 4) constant quaternion (global rotation) at each step
    sm_trans  : (N, 3) smoothed per-step translations
    R_global  : (3, 3) the single global rotation matrix
    """
    N = len(src_all)
    valid_idx = np.where(valid_mask)[0]
    src_scaled = src_all * fixed_scale
    src_v  = src_scaled[valid_idx]
    dst_v  = dst_all[valid_idx]
    n_valid = len(valid_idx)

    head_valid_idx  = np.where(head_valid)[0] if head_valid is not None else None
    head_src_scaled = head_src_all * fixed_scale if head_src_all is not None else None

    # 1. Single global rotation from all valid pairs (heel + head)
    aug_src, aug_dst = _augment_with_head(
        src_v, dst_v, head_src_scaled, head_dst_all, head_valid_idx, 0, N)
    R_global, _ = umeyama_rigid(aug_src, aug_dst)

    # 2. Pre-rotate every source point
    src_rot_all = (R_global @ src_scaled.T).T   # (N, 3)
    src_rot_v   = src_rot_all[valid_idx]
    # Pre-rotate head src for translation window
    head_src_rot = (R_global @ head_src_scaled.T).T if head_src_scaled is not None else None

    half_w    = window_size // 2
    raw_trans = np.zeros((N, 3))

    # 3. Per-step local window → mean translation only
    for i in range(N):
        pos = int(np.clip(np.searchsorted(valid_idx, i), 0, n_valid - 1))
        lo  = max(0, pos - half_w)
        hi  = min(n_valid, lo + window_size)
        lo  = max(0, hi - window_size)

        w_src = src_rot_v[lo:hi]
        w_dst = dst_v[lo:hi]

        # Append head translation contributions in this step range
        if head_valid_idx is not None and lo < n_valid:
            step_lo = int(valid_idx[lo])
            step_hi = int(valid_idx[min(hi, n_valid) - 1]) + 1
            h_in = head_valid_idx[(head_valid_idx >= step_lo) & (head_valid_idx < step_hi)]
            if len(h_in):
                w_src = np.vstack([w_src, head_src_rot[h_in]])
                w_dst = np.vstack([w_dst, head_dst_all[h_in]])

        if len(w_src) >= 1:
            raw_trans[i] = (w_dst - w_src).mean(axis=0)
        else:
            raw_trans[i] = (dst_v - src_rot_v).mean(axis=0)

    # 4. Smooth translations
    if smooth_sigma > 0:
        sm_trans = gaussian_filter1d(raw_trans, smooth_sigma, axis=0)
    else:
        sm_trans = raw_trans

    q_global = Rotation.from_matrix(R_global).as_quat()
    sm_quats = np.tile(q_global, (N, 1))

    aligned = src_rot_all + sm_trans
    return aligned, sm_quats, sm_trans, R_global


def windowed_translation_global_rot_adaptive(
    src_all, dst_all, valid_mask,
    base_window=60,
    min_window=8,
    residual_threshold_factor=1.5,
    smooth_sigma=5.0,
    fixed_scale=0.9,
    head_src_all=None, head_dst_all=None, head_valid=None,
):
    """
    Two-pass adaptive windowed translation with one global rotation.

    Identical to windowed_rigid_adaptive but the rotation is fixed globally;
    only the per-window translation is optimised.

    Optional Aria head correspondences augment both the global rotation fit
    and each per-step translation window.

    Parameters
    ----------
    src_all                  : (N, 3) Shadow heel positions
    dst_all                  : (N, 3) DJI heel positions (NaN where missing)
    valid_mask               : (N,) bool
    base_window              : int   – largest window (low-residual regions)
    min_window               : int   – smallest window (high-residual regions)
    residual_threshold_factor: float – threshold = mean + k * std
    smooth_sigma             : float – Gaussian σ for smoothing translations
    fixed_scale              : float – pre-scaling applied to src
    head_src_all : (N, 3) or None – shadow Head positions
    head_dst_all : (N, 3) or None – Aria camera positions in DJI world
    head_valid   : (N,) bool or None

    Returns
    -------
    aligned         : (N, 3) aligned Shadow positions
    sm_quats        : (N, 4) constant quaternion (global rotation) at each step
    sm_trans        : (N, 3) smoothed per-step translations
    adaptive_windows: (N,)   int, window size used per step
    residuals_pass1 : (N,)   float, per-step residual from pass 1
    R_global        : (3, 3) the single global rotation matrix
    """
    N = len(src_all)
    valid_idx = np.where(valid_mask)[0]
    src_scaled = src_all * fixed_scale
    src_v  = src_scaled[valid_idx]
    dst_v  = dst_all[valid_idx]
    n_valid = len(valid_idx)

    head_valid_idx  = np.where(head_valid)[0] if head_valid is not None else None
    head_src_scaled = head_src_all * fixed_scale if head_src_all is not None else None

    # Single global rotation (augmented with head pairs)
    aug_src, aug_dst = _augment_with_head(
        src_v, dst_v, head_src_scaled, head_dst_all, head_valid_idx, 0, N)
    R_global, _ = umeyama_rigid(aug_src, aug_dst)
    src_rot_all = (R_global @ src_scaled.T).T   # (N, 3)
    src_rot_v   = src_rot_all[valid_idx]
    head_src_rot = (R_global @ head_src_scaled.T).T if head_src_scaled is not None else None

    # ── Pass 1: broad window → per-valid-point residuals ─────────────────────
    print(f"  [global-rot adaptive] Pass 1: base_window={base_window} …", flush=True)
    residuals_v = np.zeros(n_valid)
    for vi in range(n_valid):
        half_w = base_window // 2
        lo = max(0, vi - half_w)
        hi = min(n_valid, lo + base_window)
        lo = max(0, hi - base_window)
        t    = (dst_v[lo:hi] - src_rot_v[lo:hi]).mean(axis=0)
        pred = src_rot_v[vi] + t
        residuals_v[vi] = np.linalg.norm(pred - dst_v[vi])

    residuals_all = np.interp(np.arange(N), valid_idx, residuals_v)

    r_mean    = residuals_all.mean()
    r_std     = residuals_all.std()
    threshold = r_mean + residual_threshold_factor * r_std
    print(f"  [global-rot adaptive] Residual pass-1:  mean={r_mean:.4f}  "
          f"std={r_std:.4f}  threshold={threshold:.4f}", flush=True)

    # ── Adaptive window sizes ──────────────────────────────────────────────────
    t_val = np.clip((residuals_all - r_mean) / (threshold - r_mean + 1e-9), 0.0, 1.0)
    adaptive_windows_raw = np.round(
        base_window - t_val * (base_window - min_window)
    ).astype(int)
    adaptive_windows_raw = np.clip(adaptive_windows_raw, min_window, base_window)

    if smooth_sigma > 0:
        adaptive_windows = np.round(
            gaussian_filter1d(adaptive_windows_raw.astype(float), smooth_sigma)
        ).astype(int)
        adaptive_windows = np.clip(adaptive_windows, min_window, base_window)
    else:
        adaptive_windows = adaptive_windows_raw

    n_tight = int((adaptive_windows < base_window * 0.8).sum())
    print(f"  [global-rot adaptive] Tight windows (<{int(base_window*0.8)} steps): "
          f"{n_tight}/{N} steps  "
          f"range=[{adaptive_windows.min()}, {adaptive_windows.max()}]", flush=True)

    # ── Pass 2: per-step adaptive translation ─────────────────────────────────
    print(f"  [global-rot adaptive] Pass 2: per-step adaptive translation …", flush=True)
    raw_trans = np.zeros((N, 3))
    for i in range(N):
        pos = int(np.clip(np.searchsorted(valid_idx, i), 0, n_valid - 1))
        w      = int(adaptive_windows[i])
        half_w = w // 2
        lo = max(0, pos - half_w)
        hi = min(n_valid, lo + w)
        lo = max(0, hi - w)

        w_src = src_rot_v[lo:hi]
        w_dst = dst_v[lo:hi]

        # Append head translation contributions in this step range
        if head_valid_idx is not None and lo < n_valid:
            step_lo = int(valid_idx[lo])
            step_hi = int(valid_idx[min(hi, n_valid) - 1]) + 1
            h_in = head_valid_idx[(head_valid_idx >= step_lo) & (head_valid_idx < step_hi)]
            if len(h_in):
                w_src = np.vstack([w_src, head_src_rot[h_in]])
                w_dst = np.vstack([w_dst, head_dst_all[h_in]])

        if len(w_src) >= 1:
            raw_trans[i] = (w_dst - w_src).mean(axis=0)
        else:
            raw_trans[i] = (dst_v - src_rot_v).mean(axis=0)

    # ── Smooth translations ────────────────────────────────────────────────────
    if smooth_sigma > 0:
        sm_trans = gaussian_filter1d(raw_trans, smooth_sigma, axis=0)
    else:
        sm_trans = raw_trans

    q_global = Rotation.from_matrix(R_global).as_quat()
    sm_quats  = np.tile(q_global, (N, 1))

    aligned = src_rot_all + sm_trans
    return aligned, sm_quats, sm_trans, adaptive_windows, residuals_all, R_global


# ── Top-down 2-D matplotlib plot ──────────────────────────────────────────────

def plot_topdown_2d(
    aligned,
    dst_all,
    valid_mask,
    window_labels,
    mode_label="",
    breakpoints=None,
    heading_deg=None,
    angular_delta=None,
    save_path="topdown_2d.png",
):
    """
    Save a top-down (X-Y) view of the aligned trajectory coloured by
    rotation-detected window, with DJI hit points overlaid.

    An optional inset axes shows the per-step angular delta and detected
    breakpoints if heading/delta data are provided.

    Parameters
    ----------
    aligned       : (N, 3)
    dst_all       : (N, 3)  NaN where missing
    valid_mask    : (N,) bool
    window_labels : (N,) int – window index per step (drives colouring)
    mode_label    : str  – subtitle text (alignment mode description)
    breakpoints   : list[int] or None
    heading_deg   : (N,) float or None
    angular_delta : (N,) float or None
    save_path     : str  – output PNG path
    """
    N          = len(aligned)
    n_windows  = int(window_labels.max()) + 1
    breakpoints = breakpoints or [0]

    # ── Colour palette ─────────────────────────────────────────────────────────
    # Use tab20 for up to 20 windows; cycle if more.
    try:
        palette = matplotlib.colormaps.get_cmap("tab20").resampled(max(n_windows, 2))
    except AttributeError:                      # matplotlib < 3.7 fallback
        palette = cm.get_cmap("tab20", max(n_windows, 2))
    win_colors = [palette(w % 20) for w in range(n_windows)]

    # ── Figure layout ─────────────────────────────────────────────────────────
    has_inset = heading_deg is not None and angular_delta is not None
    if has_inset:
        fig = plt.figure(figsize=(14, 10))
        ax_main = fig.add_axes([0.07, 0.28, 0.88, 0.66])   # main top-down
        ax_ang  = fig.add_axes([0.07, 0.06, 0.88, 0.18])   # angular delta strip
    else:
        fig, ax_main = plt.subplots(figsize=(13, 10))

    # ── Main top-down plot ────────────────────────────────────────────────────
    legend_handles = []

    for w in range(n_windows):
        mask = window_labels == w
        pts  = aligned[mask]
        if len(pts) == 0:
            continue
        color = win_colors[w]
        # Trajectory line
        ax_main.plot(pts[:, 0], pts[:, 1],
                     '-', color=color, alpha=0.6, linewidth=1.5, zorder=2)
        # Step dots
        ax_main.scatter(pts[:, 0], pts[:, 1],
                        color=color, s=8, alpha=0.7, zorder=3)
        legend_handles.append(
            mpatches.Patch(color=color, label=f"Window {w}  ({mask.sum()} steps)")
        )

    # Window-boundary markers
    for bp in breakpoints[1:]:
        if bp < N:
            ax_main.axvline(x=aligned[bp, 0], color="white", alpha=0.25,
                            linewidth=0.8, linestyle="--", zorder=1)
            ax_main.annotate(
                f"W{window_labels[bp]}",
                xy=(aligned[bp, 0], aligned[bp, 1]),
                fontsize=7, color="white", ha="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45),
                zorder=6,
            )

    # DJI ground-contact hits
    dst_valid = dst_all[valid_mask]
    ax_main.scatter(
        dst_valid[:, 0], dst_valid[:, 1],
        c="gold", s=35, zorder=5, marker="*",
        edgecolors="darkorange", linewidths=0.5, label="DJI hits",
    )
    legend_handles.append(
        mpatches.Patch(color="gold", label=f"DJI hits  ({dst_valid.shape[0]})")
    )

    ax_main.set_xlabel("X  (m)", fontsize=10)
    ax_main.set_ylabel("Y  (m)", fontsize=10)
    ax_main.set_title(
        f"Top-down trajectory — {mode_label}\n"
        f"Coloured by rotation-detected window  ({n_windows} windows)",
        fontsize=12,
    )
    ax_main.legend(handles=legend_handles, loc="upper right",
                   fontsize=7, ncol=max(1, n_windows // 10 + 1))
    ax_main.set_aspect("equal", adjustable="datalim")
    ax_main.set_facecolor("#0a0a10")
    ax_main.grid(True, color="gray", alpha=0.2, linewidth=0.5)

    # ── Angular-delta inset ───────────────────────────────────────────────────
    if has_inset:
        steps = np.arange(N)
        ax_ang.fill_between(steps, angular_delta, alpha=0.5, color="steelblue",
                            label="Angular Δ (deg)")
        ax_ang.plot(steps, angular_delta, color="steelblue", linewidth=0.8)

        # Shade windows
        for w in range(n_windows):
            start = breakpoints[w]
            end   = breakpoints[w + 1] if w + 1 < len(breakpoints) else N
            ax_ang.axvspan(start, end, alpha=0.15, color=win_colors[w], zorder=0)

        # Mark breakpoints
        for bp in breakpoints[1:]:
            ax_ang.axvline(bp, color="white", linewidth=0.8, linestyle="--", alpha=0.5)

        # Threshold line
        if angular_delta.max() > 0:
            ax_ang.set_xlim(0, N - 1)
            ax_ang.set_ylim(bottom=0)

        ax_ang.set_xlabel("Step index", fontsize=9)
        ax_ang.set_ylabel("Heading Δ (°)", fontsize=9)
        ax_ang.set_title("Per-step heading change — window boundaries marked",
                         fontsize=9)
        ax_ang.set_facecolor("#0a0a10")
        ax_ang.tick_params(colors="gray")
        ax_ang.spines[:].set_color("gray")
        ax_ang.grid(True, color="gray", alpha=0.15, linewidth=0.5)

    fig.patch.set_facecolor("#06060c")
    for a in fig.get_axes():
        a.tick_params(colors="lightgray", labelsize=8)
        for spine in a.spines.values():
            spine.set_color("#444")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved 2D top-down plot → {save_path}")


# ── Apply alignment to detected_steps.csv ────────────────────────────────────


# ── PLY loader ────────────────────────────────────────────────────────────────

def load_ply_vertices(path, n_sample):
    print("Reading PLY header …", flush=True)
    with open(path, "rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.strip() == b"end_header":
                break
        header_text = header_bytes.decode("utf-8", errors="replace")
        n_verts = 0
        for ln in header_text.splitlines():
            if ln.startswith("element vertex"):
                n_verts = int(ln.split()[-1])
                break
        print(f"  {n_verts:,} vertices — sampling {n_sample:,} …", flush=True)
        vertex_bytes = f.read(n_verts * 6 * 4)

    arr = np.frombuffer(vertex_bytes, dtype="<f4").reshape(n_verts, 6)
    rng = np.random.default_rng(42)
    idx = rng.choice(n_verts, size=min(n_sample, n_verts), replace=False)
    xyz = arr[idx, :3].astype(np.float64)
    rgb = arr[idx, 3:6].astype(np.float64)
    if rgb.max() > 1.01:
        rgb /= 255.0
    return xyz, np.clip(rgb, 0.0, 1.0)


# ── Sphere helper ─────────────────────────────────────────────────────────────

def make_sphere(centre, radius, color):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    mesh.translate(centre)
    return mesh


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Windowed Umeyama heel alignment")

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--fixed-window", action="store_true",
                            help="Use original fixed-window mode")
    mode_group.add_argument("--rotation-window", action="store_true",
                            help="Use rotation-based windowing mode")
    # (default when neither flag given → adaptive)

    # ── Fixed-window args ─────────────────────────────────────────────────────
    parser.add_argument("--window-size",  type=int,   default=30,
                        help="[fixed mode] Matched pairs per local fit (default 30)")

    # ── Adaptive-window args ───────────────────────────────────────────────────
    parser.add_argument("--base-window",  type=int,   default=60,
                        help="[adaptive] Max window (low-residual zones, default 60)")
    parser.add_argument("--min-window",   type=int,   default=8,
                        help="[adaptive] Min window (high-residual zones, default 8)")
    parser.add_argument("--residual-threshold", type=float, default=1.5,
                        help="[adaptive] Threshold = mean + K*std  (default 1.5)")

    # ── Rotation-window args ──────────────────────────────────────────────────
    parser.add_argument("--rotation-threshold", type=float, default=30.0,
                        help="[rotation] Heading change (deg) to start a new window "
                             "(default 30)")
    parser.add_argument("--rotation-min-steps", type=int, default=20,
                        help="[rotation] Minimum steps per window (default 20)")
    parser.add_argument("--heading-smooth", type=float, default=5.0,
                        help="[rotation] Gaussian σ for smoothing trajectory before "
                             "heading computation (default 5)")

    # ── Shared ────────────────────────────────────────────────────────────────
    parser.add_argument("--smooth-sigma", type=float, default=5.0,
                        help="Gaussian smoothing σ in steps (default 5, 0=off)")
    parser.add_argument("--global-rotation", action="store_true",
                        help="Use a single global rotation fitted from all valid pairs; "
                             "per-window translation is still computed locally. "
                             "Compatible with fixed-window and adaptive modes.")
    parser.add_argument("--no-2d-plot", action="store_true",
                        help="Skip saving the top-down 2D matplotlib figure")
    parser.add_argument("--no-aria", action="store_true",
                        help="Disable Aria head-position constraint "
                             "(use heel-only alignment as before)")

    # ── Required path arguments ───────────────────────────────────────────────
    parser.add_argument("--heels-csv", required=True,
                        help="Input median heel positions CSV "
                             "(e.g. median_heel_positions.csv)")
    parser.add_argument("--ply", required=True,
                        help="DJI reconstruction PLY mesh for visualisation "
                             "(e.g. dji/recon_1.ply)")
    parser.add_argument("--out-csv", required=True,
                        help="Output aligned heel positions CSV "
                             "(e.g. median_heel_positions_aligned.csv)")
    parser.add_argument("--out-npz", required=True,
                        help="Output Umeyama transform NPZ "
                             "(e.g. umeyama_transform.npz)")
    parser.add_argument("--plot-path", required=True,
                        help="Output top-down 2D plot PNG "
                             "(e.g. topdown_2d.png)")

    # ── Optional Aria path arguments (skipped with warning if absent) ─────────
    parser.add_argument("--detected-steps-csv",
                        help="detected_steps.csv from detect_steps.py "
                             "(required for Aria head constraint)")
    parser.add_argument("--frame-positions",
                        help="aria/frame_positions.json "
                             "(required for Aria head constraint)")
    parser.add_argument("--dji-aria-matches",
                        help="dji_aria_frame_matches.csv "
                             "(required for Aria head constraint)")
    parser.add_argument("--time-aligned",
                        help="time_aligned_steps.csv "
                             "(required for Aria head constraint)")
    parser.add_argument("--alignment-npz",
                        help="alignment_transform.npz "
                             "(required for Aria head constraint)")

    args = parser.parse_args()

    smooth_sigma = args.smooth_sigma

    global_rotation = args.global_rotation

    if args.fixed_window:
        mode = "fixed"
        window_size = args.window_size
        print(f"Mode: fixed-window  window_size={window_size}  smooth_sigma={smooth_sigma}"
              + ("  [global-rotation]" if global_rotation else ""))
    elif args.rotation_window:
        mode = "rotation"
        rot_thresh     = args.rotation_threshold
        rot_min_steps  = args.rotation_min_steps
        heading_smooth = args.heading_smooth
        if global_rotation:
            print("Warning: --global-rotation is not supported in rotation-window mode; "
                  "ignoring flag.")
            global_rotation = False
        print(f"Mode: rotation-window  threshold={rot_thresh}°  "
              f"min_steps={rot_min_steps}  heading_smooth={heading_smooth}  "
              f"smooth_sigma={smooth_sigma}")
    else:
        mode = "adaptive"
        base_window  = args.base_window
        min_window   = args.min_window
        res_thresh   = args.residual_threshold
        print(f"Mode: adaptive  base_window={base_window}  min_window={min_window}  "
              f"residual_threshold={res_thresh}  smooth_sigma={smooth_sigma}"
              + ("  [global-rotation]" if global_rotation else ""))

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(args.heels_csv)
    N  = len(df)

    src_all   = df[["shadow_heel_x", "shadow_heel_y", "shadow_heel_z"]].values
    dst_raw   = df[["median_hit_x",  "median_hit_y",  "median_hit_z" ]].values
    valid     = ~np.isnan(dst_raw[:, 0])
    dst_all   = dst_raw.copy()

    print(f"  {N} steps  ({valid.sum()} with valid DJI hits)")

    # ── Load Aria head correspondences ────────────────────────────────────────
    if args.no_aria:
        head_src_all = head_dst_all = None
        head_valid   = None
        print("  [aria] Head constraint disabled (--no-aria)")
    else:
        print("\nLoading Aria head positions …")
        head_src_all, head_dst_all, head_valid = load_aria_head_positions(
            df,
            alignment_npz_path=args.alignment_npz,
            frame_positions_path=args.frame_positions,
            dji_aria_matches_path=args.dji_aria_matches,
            time_aligned_path=args.time_aligned,
            detected_steps_csv=args.detected_steps_csv,
        )
    head_kwargs = dict(
        head_src_all=head_src_all,
        head_dst_all=head_dst_all,
        head_valid=head_valid,
    )

    # ── Global Umeyama (reference) ────────────────────────────────────────────
    R_g, t_g, c_g = umeyama(src_all[valid], dst_all[valid])
    global_aligned = apply_transform(src_all, R_g, t_g, c_g)
    res_g = np.linalg.norm(global_aligned[valid] - dst_all[valid], axis=1)
    print(f"\nGlobal Umeyama  RMSE={np.sqrt((res_g**2).mean()):.4f} m  "
          f"scale={c_g:.4f}")

    # ── Always detect rotation windows (used for 2-D plot in every mode) ───────
    print(f"\nDetecting rotation windows (for 2D visualisation) …")
    rot_thresh_vis    = args.rotation_threshold
    rot_min_steps_vis = args.rotation_min_steps
    heading_smooth_vis= args.heading_smooth
    vis_window_labels, vis_breakpoints, vis_heading, vis_delta = \
        detect_rotation_windows(
            src_all,
            threshold_deg=rot_thresh_vis,
            min_window_steps=rot_min_steps_vis,
            heading_smooth_sigma=heading_smooth_vis,
        )

    # ── Windowed + smoothed alignment ─────────────────────────────────────────
    if mode == "fixed":
        if global_rotation:
            print(f"\nComputing fixed-window global-rotation alignment "
                  f"(W={window_size}, σ={smooth_sigma}) …")
            aligned, sm_quats, sm_trans, R_global_used = \
                windowed_translation_global_rot(
                    src_all, dst_all, valid, window_size, smooth_sigma,
                    **head_kwargs
                )
            res_w = np.linalg.norm(aligned[valid] - dst_all[valid], axis=1)
            print(f"Fixed-window (global-rot) RMSE={np.sqrt((res_w**2).mean()):.4f} m")
            sm_scales        = np.ones(N) * 0.9
            adaptive_windows = np.full(N, window_size)
            residuals_pass1  = res_w
        else:
            print(f"\nComputing fixed-window alignment (W={window_size}, σ={smooth_sigma}) …")
            aligned, sm_quats, sm_trans, sm_scales = windowed_rigid(
                src_all, dst_all, valid, window_size, smooth_sigma,
                **head_kwargs
            )
            res_w = np.linalg.norm(aligned[valid] - dst_all[valid], axis=1)
            print(f"Fixed-window RMSE={np.sqrt((res_w**2).mean()):.4f} m  "
                  f"scale range=[{sm_scales.min():.4f}, {sm_scales.max():.4f}]")
            adaptive_windows = np.full(N, window_size)
            residuals_pass1  = res_w  # same-pass, used only for saving

    elif mode == "rotation":
        print(f"\nComputing rotation-window alignment …")
        aligned, sm_quats, sm_trans, _ = windowed_by_rotation(
            src_all, dst_all, valid,
            window_labels=vis_window_labels,
            smooth_sigma=smooth_sigma,
            **head_kwargs
        )
        res_w = np.linalg.norm(aligned[valid] - dst_all[valid], axis=1)
        sm_scales = np.ones(N) * 0.9
        adaptive_windows = vis_window_labels  # repurpose field for CSV output
        residuals_pass1  = res_w
        n_rot_windows    = int(vis_window_labels.max()) + 1
        print(f"Rotation-window RMSE={np.sqrt((res_w**2).mean()):.4f} m  "
              f"({n_rot_windows} windows)")

    else:  # adaptive
        if global_rotation:
            print(f"\nComputing adaptive-window global-rotation alignment …")
            aligned, sm_quats, sm_trans, adaptive_windows, residuals_pass1, R_global_used = \
                windowed_translation_global_rot_adaptive(
                    src_all, dst_all, valid,
                    base_window=base_window,
                    min_window=min_window,
                    residual_threshold_factor=res_thresh,
                    smooth_sigma=smooth_sigma,
                    **head_kwargs
                )
            res_w = np.linalg.norm(aligned[valid] - dst_all[valid], axis=1)
            sm_scales = np.ones(N) * 0.9
            print(f"Adaptive-window (global-rot) RMSE={np.sqrt((res_w**2).mean()):.4f} m")
        else:
            print(f"\nComputing adaptive-window alignment …")
            aligned, sm_quats, sm_trans, adaptive_windows, residuals_pass1 = \
                windowed_rigid_adaptive(
                    src_all, dst_all, valid,
                    base_window=base_window,
                    min_window=min_window,
                    residual_threshold_factor=res_thresh,
                    smooth_sigma=smooth_sigma,
                    **head_kwargs
                )
            res_w = np.linalg.norm(aligned[valid] - dst_all[valid], axis=1)
            sm_scales = np.ones(N) * 0.9
            print(f"Adaptive-window RMSE={np.sqrt((res_w**2).mean()):.4f} m")

        # Print a small report of where windows were tightened most
        tight_thresh = int(base_window * 0.8)
        tight_steps  = np.where(adaptive_windows < tight_thresh)[0]
        if len(tight_steps):
            # Find contiguous runs of tight steps
            breaks = np.where(np.diff(tight_steps) > 5)[0] + 1
            runs   = np.split(tight_steps, breaks)
            print(f"\n  Tight-window zones (window < {tight_thresh} steps):")
            for run in runs[:20]:   # cap at 20 zones
                w_min = adaptive_windows[run].min()
                print(f"    steps {run[0]:4d}–{run[-1]:4d}  "
                      f"({len(run):3d} steps)  "
                      f"min_window={w_min}")
            if len(runs) > 20:
                print(f"    … and {len(runs)-20} more zones")
        else:
            print("  No tight-window zones detected.")

    # ── Aria head alignment quality report ───────────────────────────────────
    head_aligned = None   # populated below if Aria data is available
    if head_valid is not None and head_valid.any():
        # Apply current per-step transform to shadow Head positions
        head_aligned = np.zeros((N, 3))
        for i in range(N):
            R_i = Rotation.from_quat(sm_quats[i]).as_matrix()
            head_aligned[i] = apply_transform_pre_scaled(
                head_src_all[[i]] * 0.9, R_i, sm_trans[i])[0]
        head_err = np.linalg.norm(head_aligned[head_valid] - head_dst_all[head_valid], axis=1)
        print(f"\nAria head alignment (on {head_valid.sum()} matched steps):")
        print(f"  RMSE = {np.sqrt((head_err**2).mean()):.4f} m  "
              f"mean = {head_err.mean():.4f} m  "
              f"max = {head_err.max():.4f} m")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_df = df.copy()
    out_df["aligned_shadow_x"]   = aligned[:, 0]
    out_df["aligned_shadow_y"]   = aligned[:, 1]
    out_df["aligned_shadow_z"]   = aligned[:, 2]
    out_df["adaptive_window"]    = adaptive_windows
    out_df["rotation_window"]    = vis_window_labels
    out_df.to_csv(args.out_csv, index=False)

    np.savez(args.out_npz, R=R_g, t=t_g, c=np.array(c_g),
             sm_quats=sm_quats, sm_trans=sm_trans, sm_scales=sm_scales,
             adaptive_windows=adaptive_windows,
             rotation_window_labels=vis_window_labels,
             smooth_sigma=np.array(smooth_sigma))

    # ── Apply alignment to detected_steps.csv ────────────────────────────────
    import os
    # ── Top-down 2D matplotlib plot ───────────────────────────────────────────
    if not args.no_2d_plot:
        print(f"\nGenerating top-down 2D plot …")
        if mode == "fixed":
            mode_label_2d = (f"fixed-window  W={window_size}  σ={smooth_sigma}"
                             + ("  global-rot" if global_rotation else ""))
        elif mode == "rotation":
            mode_label_2d = (f"rotation-window  thresh={rot_thresh_vis}°  "
                             f"min_steps={rot_min_steps_vis}  σ={smooth_sigma}")
        else:
            mode_label_2d = (f"adaptive-window  base={base_window}  "
                             f"min={min_window}  σ={smooth_sigma}"
                             + ("  global-rot" if global_rotation else ""))
        plot_topdown_2d(
            aligned, dst_all, valid,
            window_labels=vis_window_labels,
            mode_label=mode_label_2d,
            breakpoints=vis_breakpoints,
            heading_deg=vis_heading,
            angular_delta=vis_delta,
            save_path=args.plot_path,
        )

    # ── Load PLY ──────────────────────────────────────────────────────────────
    xyz, rgb = load_ply_vertices(args.ply, N_PLY_SAMPLE)
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(xyz)
    mesh_pcd.colors = o3d.utility.Vector3dVector(rgb * 0.7)

    # ── Build geometry ────────────────────────────────────────────────────────
    dst_valid = dst_all[valid]
    win_valid = aligned[valid]
    glo_valid = global_aligned[valid]

    res_w_v = np.linalg.norm(win_valid - dst_valid, axis=1)
    res_max  = res_w_v.max() if res_w_v.max() > 0 else 1.0

    SPHERE_R = 0.04

    geoms = [mesh_pcd]

    # DJI hits – yellow
    for pt in dst_valid:
        geoms.append(make_sphere(pt, SPHERE_R, [1.0, 0.85, 0.0]))

    # Windowed-aligned shadow – cyan (matched), magenta (no hit)
    # Adaptive mode: tight-window steps get a lime outline sphere.
    # Rotation mode: each window gets its own colour from a palette.
    tight_thresh_vis = int(base_window * 0.8) if mode == "adaptive" else -1
    import matplotlib as _mpl
    _n_rw = max(int(vis_window_labels.max()) + 1, 2)
    try:
        _rot_palette = _mpl.colormaps.get_cmap("tab20").resampled(_n_rw)
    except AttributeError:
        _rot_palette = _mpl.cm.get_cmap("tab20", _n_rw)

    for i in range(N):
        if mode == "rotation":
            # Colour by rotation window; dim when no valid DJI hit
            w_col = _rot_palette(int(vis_window_labels[i]) % 20)[:3]
            color = list(w_col) if valid[i] else [c * 0.5 for c in w_col]
        else:
            color = [0.0, 0.9, 1.0] if valid[i] else [1.0, 0.3, 1.0]
        geoms.append(make_sphere(aligned[i], SPHERE_R, color))
        if mode == "adaptive" and int(adaptive_windows[i]) < tight_thresh_vis:
            # Larger translucent sphere in lime to mark tight-window zone
            geoms.append(make_sphere(aligned[i], SPHERE_R * 2.0, [0.3, 1.0, 0.3]))

    # Residual lines for matched pairs (windowed): green→red by residual
    line_pts, line_segs, line_colors = [], [], []
    vi = 0
    for i in range(N):
        if not valid[i]:
            continue
        d = dst_all[i];  w = aligned[i]
        t_val = float(np.clip(res_w_v[vi] / res_max, 0, 1))
        i0 = len(line_pts); line_pts.append(d)
        i1 = len(line_pts); line_pts.append(w)
        line_segs.append([i0, i1])
        line_colors.append([t_val, 1.0 - t_val, 0.0])
        vi += 1

    connector = o3d.geometry.LineSet()
    connector.points = o3d.utility.Vector3dVector(np.array(line_pts))
    connector.lines  = o3d.utility.Vector2iVector(np.array(line_segs))
    connector.colors = o3d.utility.Vector3dVector(np.array(line_colors))
    geoms.append(connector)

    # ── Aria head geometry ────────────────────────────────────────────────────
    head_vis_active = head_aligned is not None and head_valid is not None
    if head_vis_active:
        n_head = int(head_valid.sum())
        head_err_v = np.linalg.norm(
            head_aligned[head_valid] - head_dst_all[head_valid], axis=1)
        head_err_max = head_err_v.max() if head_err_v.max() > 0 else 1.0

        HEAD_R       = SPHERE_R * 1.1
        h_line_pts, h_line_segs, h_line_colors = [], [], []
        hi_vi = 0

        for i in range(N):
            # Aligned shadow Head – orange (matched to Aria) or dim orange (unmatched)
            if head_valid[i]:
                geoms.append(make_sphere(head_aligned[i], HEAD_R, [1.0, 0.55, 0.1]))
            else:
                geoms.append(make_sphere(head_aligned[i], HEAD_R, [0.5, 0.28, 0.05]))

            if not head_valid[i]:
                continue

            # Aria camera position – lime green sphere
            aria_pt = head_dst_all[i]
            geoms.append(make_sphere(aria_pt, HEAD_R, [0.2, 1.0, 0.3]))

            # Line: aligned head → Aria camera, coloured blue→red by error
            t_h = float(np.clip(head_err_v[hi_vi] / head_err_max, 0, 1))
            i0 = len(h_line_pts); h_line_pts.append(head_aligned[i])
            i1 = len(h_line_pts); h_line_pts.append(aria_pt)
            h_line_segs.append([i0, i1])
            h_line_colors.append([t_h, 0.0, 1.0 - t_h])   # blue→red
            hi_vi += 1

        if h_line_pts:
            head_conn = o3d.geometry.LineSet()
            head_conn.points = o3d.utility.Vector3dVector(np.array(h_line_pts))
            head_conn.lines  = o3d.utility.Vector2iVector(np.array(h_line_segs))
            head_conn.colors = o3d.utility.Vector3dVector(np.array(h_line_colors))
            geoms.append(head_conn)

    # ── Open viewer ───────────────────────────────────────────────────────────
    head_legend = "  Orange=head(aligned)  Lime=Aria-cam" if head_vis_active else ""
    if mode == "fixed":
        title = (f"Fixed-window Umeyama  W={window_size}  σ={smooth_sigma}"
                 + ("  GlobalRot" if global_rotation else "")
                 + "  – Yellow=DJI  Cyan=Shadow  Magenta=no hit" + head_legend)
    elif mode == "rotation":
        n_rot_w = int(vis_window_labels.max()) + 1
        title   = (f"Rotation-window Umeyama  thresh={rot_thresh_vis}°  "
                   f"min_steps={rot_min_steps_vis}  σ={smooth_sigma}  "
                   f"– Yellow=DJI  Coloured=window  Dim=no hit  "
                   f"({n_rot_w} windows)" + head_legend)
    else:
        title = (f"Adaptive-window Umeyama  base={base_window} min={min_window}  "
                 f"thresh={res_thresh}  σ={smooth_sigma}"
                 + ("  GlobalRot" if global_rotation else "")
                 + "  – Yellow=DJI  Cyan=Shadow  Magenta=no hit  "
                   "Green sphere=tight window" + head_legend)
    print(f"\nOpening viewer …")
    print(f"  Yellow      = DJI median ground-contact (heel)")
    if mode == "rotation":
        print(f"  Colours     = Shadow heel coloured by rotation window")
        print(f"  Dim         = Shadow steps with no DJI hit")
    else:
        print(f"  Cyan        = Shadow heel (aligned, σ={smooth_sigma})")
        print(f"  Magenta     = Shadow steps with no DJI hit")
    if mode == "adaptive":
        print(f"  Green sphere outline = tight-window zone (window < {int(base_window*0.8)} steps)")
    print(f"  Lines       = green (small heel residual) → red (large)")
    if head_vis_active:
        print(f"  Orange      = aligned shadow Head  ({head_valid.sum()} matched steps)")
        print(f"  Lime green  = Aria camera position in DJI world")
        print(f"  Blue→red    = head residual lines (blue=good, red=large error)")
    if not args.no_2d_plot:
        print(f"  2D plot = {args.plot_path}")
    print(f"  Q / Esc = quit")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1600, height=900)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color      = np.array([0.04, 0.04, 0.06])
    opt.point_size            = 1.2
    opt.line_width            = 2.0
    opt.show_coordinate_frame = True

    ctrl = vis.get_view_control()
    ctrl.set_lookat(dst_valid.mean(axis=0))
    ctrl.set_up([0, 0, 1])
    ctrl.set_front([0.3, -0.7, 0.6])
    ctrl.set_zoom(0.25)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()