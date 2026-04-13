#!/usr/bin/env python3
"""
deid.py — De-identification tool for images, GIFs, and videos.

Detects every person using YOLOv8 segmentation and applies a Gaussian blur
over the selected region. Supports three modes:
  body   — full-body silhouette (default)
  faces  — head / face region only
  both   — full body blurred, faces blurred with an extra stronger pass

Detection strategy (layered):
  1. YOLOv8 segmentation mask   — precise pixel-level silhouette (body mode)
  2. Padded bounding-box fill   — catches limbs / torso at frame edges
  3. Head-region estimate       — top 30 %% of each person bbox (faces mode)
  4. Haar cascade detectors     — frontal + profile face scan (faces mode)
                                  catches faces when no body is visible
  5. Temporal mask accumulation — (video/GIF) rolling OR of the last N frames
                                  so a person stays blurred when detection
                                  briefly fails mid-pan

Usage examples
--------------
  # Full body (default)
  python deid.py --input video.mp4 --output out.mp4

  # Faces only
  python deid.py --input video.mp4 --output out.mp4 --mode faces

  # Both: body blurred + faces get an extra stronger blur on top
  python deid.py --input video.mp4 --output out.mp4 --mode both

  # Moving camera, partial bodies, faces only
  python deid.py --input video.mp4 --output out.mp4 \\
      --mode faces --model yolov8m-seg.pt --imgsz 1280 --device cuda:0

Options
-------
  --input   / -i      Path to input file (image, GIF, or video)
  --output  / -o      Path to output file (same format as input)

  Mode
  --mode              What to blur: body | faces | both  (default: body)

  Detection
  --conf              Detection confidence threshold 0–1  (default: 0.20)
  --model             YOLOv8 seg weights name or local path
                      Accuracy ladder: yolov8n < yolov8s < yolov8m < yolov8l
                      (default: yolov8n-seg.pt)
  --imgsz             Inference resolution in pixels (default: 640).
                      Use 1280 for better detection of small/partial bodies.
  --device            Compute device: cpu | cuda | cuda:0 | mps  (default: auto)
  --head-fraction     Fraction of the body bbox height used as the head region
                      in faces/both modes  (default: 0.30)

  Masking
  --blur-strength     Gaussian kernel size, odd integer  (default: 51)
  --face-blur-strength  Extra blur kernel applied to the face region in 'both'
                      mode, and the sole blur in 'faces' mode  (default: 71)
  --mask-padding      Pixels to expand each detected mask outward  (default: 20)

  Video / GIF
  --temporal-frames   Frames of mask history to OR-accumulate  (default: 4)
                      Set to 0 to disable.
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ── File-type routing ──────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
GIF_EXTS   = {".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Fraction of body bounding-box height treated as the head/face region
DEFAULT_HEAD_FRACTION = 0.30

# ── Haar cascade loader ────────────────────────────────────────────────────────

def load_face_cascades() -> list:
    """
    Load OpenCV's built-in frontal-face and profile-face Haar cascades.
    Both XML files ship with every opencv-python installation.
    Returns a list of CascadeClassifier instances (may be empty on failure).
    """
    cascade_files = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_profileface.xml",
    ]
    cascades = []
    for fname in cascade_files:
        path = str(cv2.data.haarcascades) + fname
        cc = cv2.CascadeClassifier(path)
        if not cc.empty():
            cascades.append(cc)
    if not cascades:
        print(
            "Warning: no Haar cascades loaded — face detection will rely on "
            "YOLO head-region estimates only.",
            file=sys.stderr,
        )
    return cascades


# ── Mask builders ──────────────────────────────────────────────────────────────

def _run_yolo(frame_bgr, model, conf, imgsz):
    """Run YOLO inference and return the Results object."""
    return model(frame_bgr, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]


def _dilate(mask: np.ndarray, padding: int) -> np.ndarray:
    """Morphologically dilate a binary mask by `padding` pixels."""
    if padding <= 0 or mask.max() == 0:
        return mask
    ksize  = padding * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, kernel)


def build_body_mask(
    frame_bgr: np.ndarray,
    model: YOLO,
    conf: float,
    imgsz: int,
    mask_padding: int,
) -> np.ndarray:
    """
    Full-body mask: union of YOLOv8 segmentation masks + padded bounding boxes.
    """
    h, w = frame_bgr.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    results  = _run_yolo(frame_bgr, model, conf, imgsz)

    if results.masks is not None:
        for mt in results.masks.data:
            m = cv2.resize(mt.cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
            combined = np.maximum(combined, (m > 0.5).astype(np.uint8))

    if results.boxes is not None:
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            combined[
                max(0, y1 - mask_padding): min(h, y2 + mask_padding),
                max(0, x1 - mask_padding): min(w, x2 + mask_padding),
            ] = 1

    return _dilate(combined, mask_padding)


def build_face_mask(
    frame_bgr: np.ndarray,
    model: YOLO,
    conf: float,
    imgsz: int,
    mask_padding: int,
    face_cascades: list,
    head_fraction: float,
) -> np.ndarray:
    """
    Face-only mask from two complementary sources:

    1. Head-region estimate — top `head_fraction` of each YOLO person bbox.
       Works even for partial bodies where only the torso or head is visible.

    2. OpenCV Haar cascades (frontal + profile) — catches faces when no body
       is detected at all (e.g. a face peeking around a corner, or a very
       tight close-up that confuses the person detector).
    """
    h, w = frame_bgr.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    results  = _run_yolo(frame_bgr, model, conf, imgsz)

    # ── 1. Head region from YOLO body detections ───────────────────────────────
    if results.boxes is not None:
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            box_h  = max(1, y2 - y1)
            face_y2 = y1 + int(box_h * head_fraction)
            combined[
                max(0, y1  - mask_padding): min(h, face_y2 + mask_padding),
                max(0, x1  - mask_padding): min(w, x2      + mask_padding),
            ] = 1

    # ── 2. Haar cascade detections ─────────────────────────────────────────────
    if face_cascades:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        for cascade in face_cascades:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(20, 20),
            )
            if len(faces):
                for (fx, fy, fw, fh) in faces:
                    combined[
                        max(0, fy - mask_padding): min(h, fy + fh + mask_padding),
                        max(0, fx - mask_padding): min(w, fx + fw + mask_padding),
                    ] = 1

    return _dilate(combined, mask_padding)


# ── Blur application ───────────────────────────────────────────────────────────

def apply_blur(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    blur_strength: int,
) -> np.ndarray:
    """
    Replace pixels where `mask == 1` with a Gaussian-blurred version.
    Returns the frame unchanged if the mask is empty.
    """
    if mask.max() == 0:
        return frame_bgr
    k       = max(1, blur_strength) | 1
    blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
    return np.where(mask[:, :, np.newaxis], blurred, frame_bgr).astype(np.uint8)


def process_frame(
    frame_bgr: np.ndarray,
    model: YOLO,
    conf: float,
    imgsz: int,
    mask_padding: int,
    mode: str,
    blur_strength: int,
    face_blur_strength: int,
    face_cascades: list,
    head_fraction: float,
) -> tuple:
    """
    Build mask(s) for one frame and return (result_bgr, body_mask, face_mask).
    Returning the raw masks allows the temporal accumulator to work correctly.
    """
    body_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    face_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

    if mode in ("body", "both"):
        body_mask = build_body_mask(frame_bgr, model, conf, imgsz, mask_padding)

    if mode in ("faces", "both"):
        face_mask = build_face_mask(
            frame_bgr, model, conf, imgsz, mask_padding,
            face_cascades, head_fraction,
        )

    return body_mask, face_mask


def render_frame(
    frame_bgr: np.ndarray,
    body_mask: np.ndarray,
    face_mask: np.ndarray,
    mode: str,
    blur_strength: int,
    face_blur_strength: int,
) -> np.ndarray:
    """Apply blur(s) to a frame given pre-computed masks."""
    result = frame_bgr
    if mode in ("body", "both"):
        result = apply_blur(result, body_mask, blur_strength)
    if mode in ("faces", "both"):
        result = apply_blur(result, face_mask, face_blur_strength)
    return result


# ── Temporal accumulator ───────────────────────────────────────────────────────

class TemporalMaskAccumulator:
    """
    Maintains a rolling OR of the last `n_frames` binary masks.
    Keeps separate buffers for body and face masks.
    """

    def __init__(self, n_frames: int):
        self.n_frames = n_frames
        self._body_buf: deque = deque(maxlen=max(1, n_frames))
        self._face_buf: deque = deque(maxlen=max(1, n_frames))

    def update(
        self, body_mask: np.ndarray, face_mask: np.ndarray
    ) -> tuple:
        if self.n_frames == 0:
            return body_mask, face_mask
        self._body_buf.append(body_mask.astype(bool))
        self._face_buf.append(face_mask.astype(bool))
        acc_body = np.any(np.stack(list(self._body_buf), axis=0), axis=0).astype(np.uint8)
        acc_face = np.any(np.stack(list(self._face_buf), axis=0), axis=0).astype(np.uint8)
        return acc_body, acc_face


# ── Per-format processors ──────────────────────────────────────────────────────

def _make_frame_kwargs(args, model, face_cascades) -> dict:
    return dict(
        model              = model,
        conf               = args.conf,
        imgsz              = args.imgsz,
        mask_padding       = args.mask_padding,
        mode               = args.mode,
        blur_strength      = args.blur_strength,
        face_blur_strength = args.face_blur_strength,
        face_cascades      = face_cascades,
        head_fraction      = args.head_fraction,
    )


def process_image(input_path, output_path, args, model, face_cascades):
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise ValueError(f"Could not read image: {input_path}")
    fkw = _make_frame_kwargs(args, model, face_cascades)
    body_mask, face_mask = process_frame(frame, **fkw)
    result = render_frame(frame, body_mask, face_mask,
                          args.mode, args.blur_strength, args.face_blur_strength)
    cv2.imwrite(str(output_path), result)
    print(f"Saved → {output_path}")


def process_gif(input_path, output_path, args, model, face_cascades):
    gif      = Image.open(input_path)
    n_frames = getattr(gif, "n_frames", 1)
    fkw      = _make_frame_kwargs(args, model, face_cascades)
    acc      = TemporalMaskAccumulator(args.temporal_frames)

    frames_out, durations = [], []
    for i in tqdm(range(n_frames), desc="GIF frames", unit="frame"):
        gif.seek(i)
        durations.append(gif.info.get("duration", 100))
        frame_bgr = cv2.cvtColor(np.array(gif.convert("RGBA")), cv2.COLOR_RGBA2BGR)

        body_mask, face_mask = process_frame(frame_bgr, **fkw)
        body_mask, face_mask = acc.update(body_mask, face_mask)
        result_bgr  = render_frame(frame_bgr, body_mask, face_mask,
                                   args.mode, args.blur_strength, args.face_blur_strength)
        result_rgba = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGBA)
        frames_out.append(Image.fromarray(result_rgba))

    frames_out[0].save(
        output_path, save_all=True,
        append_images=frames_out[1:], loop=0, duration=durations,
    )
    print(f"Saved → {output_path}")


def process_video(input_path, output_path, args, model, face_cascades):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    fkw = _make_frame_kwargs(args, model, face_cascades)
    acc = TemporalMaskAccumulator(args.temporal_frames)

    try:
        for _ in tqdm(range(total), desc="Video frames", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break
            body_mask, face_mask = process_frame(frame, **fkw)
            body_mask, face_mask = acc.update(body_mask, face_mask)
            writer.write(render_frame(frame, body_mask, face_mask,
                                      args.mode, args.blur_strength, args.face_blur_strength))
    finally:
        cap.release()
        writer.release()

    print(f"Saved → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="De-identify images, GIFs, and videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = parser.add_argument_group("I/O")
    io.add_argument("--input",  "-i", required=True, help="Path to input file")
    io.add_argument("--output", "-o", required=True, help="Path to output file")

    md = parser.add_argument_group("Mode")
    md.add_argument("--mode", default="body", choices=["body", "faces", "both"],
                    help="What to blur: body | faces | both")

    det = parser.add_argument_group("Detection")
    det.add_argument("--conf",           type=float, default=0.20,
                     help="Detection confidence threshold (0–1)")
    det.add_argument("--model",          default="yolov8n-seg.pt",
                     help="YOLOv8 segmentation weights name or local path")
    det.add_argument("--imgsz",          type=int, default=640,
                     help="Inference resolution (use 1280 for partial bodies)")
    det.add_argument("--device",         default=None,
                     help="Compute device: cpu | cuda | cuda:0 | mps")
    det.add_argument("--head-fraction",  type=float, default=DEFAULT_HEAD_FRACTION,
                     help="Fraction of body bbox height used as the head region")

    msk = parser.add_argument_group("Masking")
    msk.add_argument("--blur-strength",       type=int, default=51,
                     help="Body Gaussian kernel size (odd integer)")
    msk.add_argument("--face-blur-strength",  type=int, default=71,
                     help="Face Gaussian kernel size — used in faces/both modes")
    msk.add_argument("--mask-padding",        type=int, default=20,
                     help="Pixels to expand each detected mask outward")

    vid = parser.add_argument_group("Video / GIF")
    vid.add_argument("--temporal-frames", type=int, default=4,
                     help="Frames of mask history to OR-accumulate (0 = off)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        sys.exit(f"Error: input file not found: {input_path}")

    ext = input_path.suffix.lower()
    if ext not in (IMAGE_EXTS | GIF_EXTS | VIDEO_EXTS):
        sys.exit(f"Unsupported file type '{ext}'.")

    print(f"Mode          : {args.mode}")
    print(f"Loading model : {args.model}")
    model = YOLO(args.model)
    if args.device:
        model.to(args.device)

    # Load Haar cascades (only needed for faces/both modes)
    face_cascades = load_face_cascades() if args.mode in ("faces", "both") else []

    if ext in IMAGE_EXTS:
        process_image(input_path, output_path, args, model, face_cascades)
    elif ext in GIF_EXTS:
        process_gif(input_path, output_path, args, model, face_cascades)
    else:
        process_video(input_path, output_path, args, model, face_cascades)


if __name__ == "__main__":
    main()
