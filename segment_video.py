"""
SAM2 Interactive Video Segmentation
=====================================
Step 1: Shows the first frame — click to pick a point on the object.
Step 2: Shows the predicted mask on that frame so you can confirm it looks right.
Step 3: Propagates the mask through the rest of the frames and saves the result.

Usage:
  python segment_video.py \
      --frames     path/to/frames/ \
      --output     sam2_output.mp4 \
      --checkpoint sam2.1_hiera_large.pt \
      --config     configs/sam2.1/sam2.1_hiera_l.yaml \
      --out_csv    bboxes.csv \
      [--fps 30]
"""

import argparse
import os
import shutil
import tempfile

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # blocking plt.show() on Windows
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
def sorted_frame_names(frames_dir: str):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        f for f in os.listdir(frames_dir)
        if os.path.splitext(f)[1].lower() in exts
    )


def stage_frames(frames_dir: str, frame_names: list) -> str:
    """
    SAM2's video loader requires filenames whose stems are plain integers
    (e.g. 000000.jpg). This function hard-links (or copies as fallback) the
    caller-sorted frames into a temp directory under integer names and returns
    that directory path. The caller is responsible for deleting it when done.
    """
    tmp = tempfile.mkdtemp(prefix="sam2_staged_")
    for idx, name in enumerate(frame_names):
        ext = os.path.splitext(name)[1]
        dst = os.path.join(tmp, f"{idx:06d}{ext}")
        src = os.path.join(frames_dir, name)
        try:
            os.link(src, dst)          # hard-link: instant, no extra disk space
        except OSError:
            shutil.copy2(src, dst)     # fallback: copy (different drive, etc.)
    return tmp


# ── Step 1: pick a point ──────────────────────────────────────────────────────

def pick_point(image: np.ndarray):
    """Show the first frame and collect point prompts from the user."""
    print("\n── Step 1: Pick your object ──────────────────────────────")
    print("  Left-click  → foreground (include)")
    print("  Right-click → background (exclude)")
    print("  Close the window when done.\n")

    points, labels = [], []

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.canvas.manager.set_window_title("Step 1 — Pick points on first frame")
    ax.imshow(image)
    ax.set_title("Left-click = include   |   Right-click = exclude   |   Close when done",
                 fontsize=11)
    ax.axis("off")
    plt.tight_layout()

    def onclick(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x, y  = int(event.xdata), int(event.ydata)
        fg    = event.button == 1
        color = "#00ff00" if fg else "#ff4444"
        marker = "*" if fg else "X"
        points.append([x, y])
        labels.append(1 if fg else 0)
        ax.plot(x, y, marker, color=color, markersize=16,
                markeredgecolor="white", markeredgewidth=1.5)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if not points:
        h, w = image.shape[:2]
        print("  No points placed — defaulting to image centre.")
        points = [[w // 2, h // 2]]
        labels = [1]

    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


# ── Step 2: confirm the mask ──────────────────────────────────────────────────

def confirm_mask(image: np.ndarray, mask: np.ndarray, points, labels) -> bool:
    """Show the predicted mask and ask the user to confirm before propagating."""
    print("\n── Step 2: Confirm the mask ──────────────────────────────")

    overlay = image.copy().astype(np.float32)
    overlay[mask] = overlay[mask] * 0.45 + np.array([0, 220, 80]) * 0.55
    overlay = overlay.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 80), 2)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title("Step 2 — Confirm mask")

    ax1.imshow(image)
    for (px, py), lb in zip(points, labels):
        c = "#00ff00" if lb == 1 else "#ff4444"
        m = "*" if lb == 1 else "X"
        ax1.plot(px, py, m, color=c, markersize=14,
                 markeredgecolor="white", markeredgewidth=1.5)
    ax1.set_title("Your points", fontsize=11)
    ax1.axis("off")

    ax2.imshow(overlay)
    ax2.set_title("Predicted mask", fontsize=11)
    ax2.axis("off")

    plt.suptitle("Close this window, then type Y to proceed or N to pick new points.",
                 fontsize=12)
    plt.tight_layout()
    plt.show()

    answer = input("  Proceed with this mask? [Y/n]: ").strip().lower()
    return answer in ("", "y", "yes")


# ── Step 3: propagate and save ────────────────────────────────────────────────

def propagate_and_save(frames_dir, frame_names, points, labels, output_path, fps, out_csv, model_cfg, checkpoint):
    print("\n── Step 3: Propagating through video ────────────────────")
    print("  Loading SAM2 Video Predictor…")

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=DEVICE)
    video_segments = {}

    # SAM2 requires integer-named frames — stage them in a temp folder
    print("  Staging frames…")
    staged_dir = stage_frames(frames_dir, frame_names)

    try:
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            state = predictor.init_state(video_path=staged_dir)
            predictor.reset_state(state)

            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )

            for frame_idx, obj_ids, mask_logits in tqdm(
                predictor.propagate_in_video(state),
                total=len(frame_names),
                desc="  Propagating",
                unit="frame",
            ):
                video_segments[frame_idx] = {
                    oid: (mask_logits[i] > 0.0).cpu().numpy()[0]
                    for i, oid in enumerate(obj_ids)
                }
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)

    print(f"  Done — {len(video_segments)} frames processed.")

    first = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    h, w  = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    bboxes = {}

    for idx, name in enumerate(tqdm(frame_names, desc="  Writing video", unit="frame")):
        frame     = cv2.imread(os.path.join(frames_dir, name))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)

        if idx in video_segments:
            bboxes[idx] = {}
            for obj_id, mask in video_segments[idx].items():
                # Mask overlay
                frame_rgb[mask] = frame_rgb[mask] * 0.45 + np.array([0, 220, 80]) * 0.55

                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                x1 = max(int(xs.min()), 0)
                y1 = max(int(ys.min()), 0)
                x2 = min(int(xs.max()), w - 1)
                y2 = min(int(ys.max()), h - 1)
                bboxes[idx][obj_id] = (x1, y1, x2, y2)

                # Draw box on frame
                out_bgr = cv2.cvtColor(frame_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.rectangle(out_bgr, (x1, y1), (x2, y2), (0, 255, 80), 2)
                frame_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        writer.write(cv2.cvtColor(frame_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"\n  Saved video → {output_path}")

    # Save bounding boxes to a CSV
    with open(out_csv, "w") as f:
        f.write("frame,obj_id,x1,y1,x2,y2\n")
        for frame_idx in sorted(bboxes):
            for obj_id, (x1, y1, x2, y2) in bboxes[frame_idx].items():
                f.write(f"{frame_idx},{obj_id},{x1},{y1},{x2},{y2}\n")
    print(f"  Saved bboxes → {out_csv}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM2 Interactive Video Segmentation")
    parser.add_argument("--frames", required=True,
                        help="Path to folder of pre-extracted frames (JPEG/PNG, sorted by name)")
    parser.add_argument("--output", required=True,
                        help="Output video path (e.g. sam2_output.mp4)")
    parser.add_argument("--fps",    type=float, default=30.0,
                        help="Frames per second for the output video (default: 30)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the SAM2 checkpoint (.pt file)")
    parser.add_argument("--config", required=True,
                        help="Path to the SAM2 model config YAML")
    parser.add_argument("--out_csv", required=True,
                        help="Output CSV file for bounding boxes (e.g. bboxes.csv)")
    args = parser.parse_args()

    if not os.path.isdir(args.frames):
        print(f"ERROR: Folder not found: {args.frames}")
        return

    frame_names = sorted_frame_names(args.frames)
    if not frame_names:
        print(f"ERROR: No JPEG/PNG frames found in {args.frames}")
        return

    print(f"Device:     {DEVICE}")
    print(f"Frames:     {len(frame_names)} found in {args.frames}")

    first_frame = np.array(
        Image.open(os.path.join(args.frames, frame_names[0])).convert("RGB")
    )

    while True:
        # Step 1 — pick points
        points, labels = pick_point(first_frame)

        # Preview mask on first frame
        print("\n  Running SAM2 on first frame to preview mask…")
        sam2_img      = build_sam2(args.config, args.checkpoint, device=DEVICE)
        img_predictor = SAM2ImagePredictor(sam2_img)
        img_predictor.set_image(first_frame)

        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            masks, scores, _ = img_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )

        best_mask = masks[np.argmax(scores)].astype(bool)

        # Step 2 — confirm
        if confirm_mask(first_frame, best_mask, points, labels):
            break
        print("\n  Restarting point selection…")

    # Step 3 — propagate
    propagate_and_save(args.frames, frame_names, points, labels, args.output, args.fps, args.out_csv, args.config, args.checkpoint)
    print("\nAll done!")


if __name__ == "__main__":
    main()
