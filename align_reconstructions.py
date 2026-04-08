#!/usr/bin/env python3
"""
align_reconstructions.py
========================
Interactive tool to align two 3D reconstructions:

  1. Pick N corresponding points on the source (DJI) reconstruction
  2. Pick the same N points (same order) on the target (Aria) reconstruction
  3. Compute a rigid Umeyama alignment  (rotation + translation, no scale)
  4. Refine with ICP
  5. Visualise and save the combined transform

Usage:
    python align_reconstructions.py [options]

    --source PATH          Source PLY  (default: dji/recon_1.ply)
    --target PATH          Target PLY  (default: aria/fused_mesh_cleaned.ply)
    --sample-source N      Points to sample from source for picking/ICP  (default: 300000)
    --sample-target N      Points to sample from target  (default: all)
    --icp-threshold T      ICP max-correspondence distance  (default: auto)
    --icp-iterations I     ICP max iterations  (default: 200)
    --no-icp               Skip ICP; only apply Umeyama alignment
    --output PATH          Where to save the transform  (default: alignment_transform.npz)

Controls in every Open3D window:
    Mouse left-drag        – rotate
    Mouse right-drag / mid – pan
    Scroll                 – zoom
    Shift + left-click     – add a correspondence point  (pick windows only)
    Shift + right-click    – undo last picked point      (pick windows only)
    Q / Esc                – close window / confirm picks
"""

import argparse
import copy
import os
import sys

import numpy as np
import open3d as o3d


# ── geometry helpers ──────────────────────────────────────────────────────────

def load_pcd(
    path: str,
    n_points: int = None,
    compute_normals: bool = True,
) -> o3d.geometry.PointCloud:
    """
    Load a PLY file (triangle mesh *or* point cloud) as an Open3D PointCloud.
    If `n_points` is given, mesh vertices are uniformly sampled down to that count.
    """
    # Resolve path: check as-is, then relative to this script's directory
    if not os.path.exists(path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.exists(alt):
            path = alt
        else:
            # Try stripping leading directory (e.g. "aria/fused..." → "fused...")
            basename = os.path.basename(path)
            alt2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), basename)
            if os.path.exists(alt2):
                print(f"  (resolved {path!r} → {alt2!r})")
                path = alt2
            else:
                print(f"ERROR: cannot find {path!r}", file=sys.stderr)
                sys.exit(1)

    print(f"  Loading {path} …", end=" ", flush=True)

    # Attempt mesh load
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) > 0:
        nv = len(mesh.vertices)
        nf = len(mesh.triangles)
        if n_points is not None and n_points < nv:
            print(f"mesh ({nv:,} verts, {nf:,} faces) → sampling {n_points:,} pts")
            pcd = mesh.sample_points_uniformly(n_points)
        else:
            print(f"mesh ({nv:,} verts, {nf:,} faces) → using all vertices")
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            if mesh.has_vertex_colors():
                pcd.colors = mesh.vertex_colors
    else:
        # Point cloud
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            print(f"ERROR: no geometry found in {path!r}", file=sys.stderr)
            sys.exit(1)
        print(f"point cloud ({len(pcd.points):,} pts)")

    if compute_normals:
        # Estimate a reasonable radius from the bounding box diagonal
        bb = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bb.get_max_bound() - bb.get_min_bound())
        radius = max(diag * 0.005, 1e-6)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)

    return pcd


# ── correspondence picking ────────────────────────────────────────────────────

def pick_points(pcd: o3d.geometry.PointCloud, window_name: str) -> list:
    """
    Open an interactive VisualizerWithEditing window.
    The user Shift+clicks to mark points; press Q when done.
    Returns a list of picked point indices into `pcd`.
    """
    print(f"\n{'─'*62}")
    print(f"  {window_name}")
    print(f"  Shift + left-click  → select a point")
    print(f"  Shift + right-click → undo last selection")
    print(f"  Q / Esc             → confirm and close")
    print(f"{'─'*62}")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name, width=1280, height=800)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    print(f"  → {len(picked)} point(s) selected: indices {picked}")
    return picked


# ── Umeyama rigid alignment ───────────────────────────────────────────────────

def umeyama_rigid(src: np.ndarray, dst: np.ndarray):
    """
    Find the rigid transform T = (R, t) that minimises  Σ ||dst_i − (R·src_i + t)||².

    Uses the SVD-based closed-form solution (Umeyama 1991), restricted to
    rotation + translation only (no scale).

    Parameters
    ----------
    src : (N, 3) float array  – source correspondence points
    dst : (N, 3) float array  – target correspondence points

    Returns
    -------
    R   : (3, 3) rotation matrix
    t   : (3,)   translation vector
    T44 : (4, 4) homogeneous transform  [R | t; 0 0 0 1]
    """
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3, \
        "src and dst must both be (N, 3)"
    n = len(src)

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c  = src - mu_src
    dst_c  = dst - mu_dst

    # Cross-covariance
    sigma = (dst_c.T @ src_c) / n          # (3, 3)
    U, _, Vt = np.linalg.svd(sigma)

    # Correct for reflections  (det = −1 → flip last column of U)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, d])

    R = U @ D @ Vt                          # (3, 3)
    t = mu_dst - R @ mu_src                 # (3,)

    T44 = np.eye(4)
    T44[:3, :3] = R
    T44[:3,  3] = t
    return R, t, T44


# ── ICP ───────────────────────────────────────────────────────────────────────

def run_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    threshold: float,
    max_iter: int = 200,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Run ICP.  Uses point-to-plane if both clouds have normals, else point-to-point.
    `init_T` is the 4×4 initial-guess transform (identity if already pre-aligned).
    """
    if source.has_normals() and target.has_normals():
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        mode_label = "point-to-plane"
    else:
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        mode_label = "point-to-point"

    print(f"  ICP mode : {mode_label}")
    print(f"  threshold: {threshold:.4f}   max_iter: {max_iter}")

    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_T, estimator,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    return result


# ── visualisation ─────────────────────────────────────────────────────────────

def visualize(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    title: str,
    src_color=(1.0, 0.55, 0.0),   # orange
    tgt_color=(0.1, 0.55, 1.0),   # blue
):
    """Colour both clouds uniformly and open a visualisation window."""
    s = copy.deepcopy(source)
    t = copy.deepcopy(target)
    s.paint_uniform_color(src_color)
    t.paint_uniform_color(tgt_color)

    print(f"\n  ▶ Displaying: {title}")
    print("    Orange = source (DJI)   Blue = target (Aria)")
    print("    Q / Esc to close\n")

    o3d.visualization.draw_geometries(
        [s, t],
        window_name=title,
        width=1280,
        height=800,
    )


def _rotation_angle_deg(R: np.ndarray) -> float:
    """Axis-angle magnitude of a rotation matrix, in degrees."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def print_transform(label: str, R: np.ndarray, t: np.ndarray):
    angle = _rotation_angle_deg(R)
    dist  = np.linalg.norm(t)
    print(f"\n  {label}")
    print(f"    Rotation angle  : {angle:.3f}°")
    print(f"    Translation norm: {dist:.6f}")
    print(f"    t = [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")


# ── auto ICP threshold ────────────────────────────────────────────────────────

def auto_threshold(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> float:
    """
    Heuristic: 1 % of the smaller bounding-box diagonal.
    Gives a threshold in the same physical units as the point clouds.
    """
    def diag(pcd):
        bb = pcd.get_axis_aligned_bounding_box()
        return float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound()))
    return 0.01 * min(diag(source), diag(target))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Align two 3D reconstructions via Umeyama (rigid) + ICP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source",         default="dji/recon_1.ply",
                        help="Source PLY file (DJI reconstruction)")
    parser.add_argument("--target",         default="aria/fused_mesh_cleaned.ply",
                        help="Target PLY file (Aria reconstruction)")
    parser.add_argument("--sample-source",  type=int, default=300_000,
                        help="Number of points to sample from source mesh")
    parser.add_argument("--sample-target",  type=int, default=None,
                        help="Number of points to sample from target mesh (None = all)")
    parser.add_argument("--icp-threshold",  type=float, default=None,
                        help="ICP max-correspondence distance (default: auto)")
    parser.add_argument("--icp-iterations", type=int, default=200,
                        help="ICP max iterations")
    parser.add_argument("--no-icp",         action="store_true",
                        help="Skip ICP; apply Umeyama alignment only")
    parser.add_argument("--output",         default="alignment_transform.npz",
                        help="Output path for the saved transform (.npz)")
    args = parser.parse_args()

    print("=" * 62)
    print("  3D Reconstruction Aligner  —  Umeyama (rigid) + ICP")
    print("=" * 62)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading reconstructions …")
    src_pcd = load_pcd(args.source, n_points=args.sample_source)
    tgt_pcd = load_pcd(args.target, n_points=args.sample_target)

    # ── 2. Pick correspondences on source ─────────────────────────────────────
    print("\n[2/5] Pick correspondence points on the SOURCE (DJI)")
    print("      Choose clearly identifiable features (corners, poles, markers).")
    print("      You will pick the SAME points in the SAME ORDER on the target next.")
    src_idx = pick_points(src_pcd, "SOURCE — DJI  |  Shift+click to pick, Q when done")

    if len(src_idx) < 3:
        print("ERROR: At least 3 correspondence points are required.", file=sys.stderr)
        sys.exit(1)

    # ── 3. Pick correspondences on target ─────────────────────────────────────
    print("\n[3/5] Pick the SAME correspondence points on the TARGET (Aria)")
    print(f"      Pick exactly {len(src_idx)} point(s) in the same order.")
    tgt_idx = pick_points(tgt_pcd, "TARGET — Aria  |  Shift+click to pick, Q when done")

    if len(tgt_idx) < 3:
        print("ERROR: At least 3 correspondence points are required.", file=sys.stderr)
        sys.exit(1)

    # Trim to matching count if user accidentally picked different numbers
    if len(src_idx) != len(tgt_idx):
        n = min(len(src_idx), len(tgt_idx))
        print(f"WARNING: Counts differ ({len(src_idx)} vs {len(tgt_idx)}); "
              f"using first {n} pairs.")
        src_idx = src_idx[:n]
        tgt_idx = tgt_idx[:n]

    src_pts = np.asarray(src_pcd.points)[src_idx]
    tgt_pts = np.asarray(tgt_pcd.points)[tgt_idx]

    print(f"\n  Correspondence pairs ({len(src_idx)}):")
    for i, (sp, tp) in enumerate(zip(src_pts, tgt_pts)):
        print(f"    {i+1:2d}  src=[{sp[0]:9.4f} {sp[1]:9.4f} {sp[2]:9.4f}]"
              f"  →  tgt=[{tp[0]:9.4f} {tp[1]:9.4f} {tp[2]:9.4f}]")

    # ── 4. Umeyama alignment ──────────────────────────────────────────────────
    print("\n[4/5] Computing Umeyama rigid alignment …")
    R, t, T_umeyama = umeyama_rigid(src_pts, tgt_pts)
    print_transform("Umeyama result", R, t)

    # Correspondence residuals after Umeyama
    aligned_src_pts = (R @ src_pts.T).T + t
    residuals = np.linalg.norm(aligned_src_pts - tgt_pts, axis=1)
    print(f"\n    Correspondence residuals after Umeyama:")
    for i, r in enumerate(residuals):
        print(f"      pair {i+1}: {r:.6f}")
    print(f"    mean = {residuals.mean():.6f}   max = {residuals.max():.6f}")

    # Apply Umeyama to source cloud
    src_umeyama = copy.deepcopy(src_pcd)
    src_umeyama.transform(T_umeyama)

    # Show Umeyama result
    visualize(src_umeyama, tgt_pcd, "After Umeyama alignment  (Q to continue)")

    if args.no_icp:
        T_total = T_umeyama
        T_icp   = np.eye(4)
        icp_fitness = None
        icp_rmse    = None
        src_final   = src_umeyama
        print("\n  ICP skipped (--no-icp).")
    else:
        # ── 5. ICP refinement ─────────────────────────────────────────────────
        print("\n[5/5] Refining with ICP …")
        threshold = args.icp_threshold
        if threshold is None:
            threshold = auto_threshold(src_umeyama, tgt_pcd)
            print(f"  Auto ICP threshold: {threshold:.6f}")

        icp_result = run_icp(
            src_umeyama, tgt_pcd,
            init_T=np.eye(4),       # source is already Umeyama-aligned
            threshold=threshold,
            max_iter=args.icp_iterations,
        )

        T_icp = icp_result.transformation
        print_transform("ICP refinement delta", T_icp[:3, :3], T_icp[:3, 3])
        print(f"\n    ICP fitness : {icp_result.fitness:.6f}")
        print(f"    ICP RMSE    : {icp_result.inlier_rmse:.6f}")

        icp_fitness = float(icp_result.fitness)
        icp_rmse    = float(icp_result.inlier_rmse)

        # Combined: T_total maps original source → target space
        T_total = T_icp @ T_umeyama

        src_final = copy.deepcopy(src_umeyama)
        src_final.transform(T_icp)

        print("\n  Combined transform matrix (Umeyama ∘ ICP):")
        print(np.array2string(T_total, precision=8, suppress_small=True))

        visualize(src_final, tgt_pcd, "Final alignment: Umeyama + ICP  (Q to close)")

    # ── Save transform ────────────────────────────────────────────────────────
    out_path = args.output
    save_dict = dict(
        T_umeyama              = T_umeyama,
        T_icp                  = T_icp,
        T_total                = T_total,
        src_correspondence_pts = src_pts,
        tgt_correspondence_pts = tgt_pts,
        source_path            = np.bytes_(args.source),
        target_path            = np.bytes_(args.target),
    )
    if icp_fitness is not None:
        save_dict["icp_fitness"] = np.float64(icp_fitness)
        save_dict["icp_rmse"]    = np.float64(icp_rmse)

    np.savez(out_path, **save_dict)

    print(f"\n  Transform saved → {out_path}")
    print("  To apply later:")
    print("    data = np.load('alignment_transform.npz')")
    print("    pcd.transform(data['T_total'])")
    print("\nDone!")


if __name__ == "__main__":
    main()