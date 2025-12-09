# realtime_count.py - Crowd counting from video/images with left-exit / right-entry vertical lines

from line_counter import CentroidTracker, extract_peaks_from_density
import os, time, argparse, csv, math, requests
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import CSRNet

# ------------------ Helpers: model load & preprocess ------------------ #
def load_weights_into_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    new_state = {}
    for k,v in ckpt.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v
    model = CSRNet().to(device)
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model

def preprocess_image_for_model(img_rgb):
    img_t = np.transpose(img_rgb, (2,0,1)).astype(np.float32)
    img_t = torch.from_numpy(img_t).unsqueeze(0)
    return img_t

def infer_count_and_map(model, img_rgb, device):
    img_t = preprocess_image_for_model(img_rgb).to(device)
    with torch.no_grad():
        den = model(img_t)
        den_up = torch.nn.functional.interpolate(den, size=(img_t.shape[2], img_t.shape[3]), mode='bilinear', align_corners=False)
        cnt = den_up.sum().item()
        den_map = den_up.squeeze().cpu().numpy()
    return cnt, den_map

# ------------------ Overlay helper ------------------ #
def save_overlay(frame_bgr, den_map, out_path, peaks=None, tracker=None, entered=None, left=None, left_x=None, right_x=None):
    den_norm = (den_map - den_map.min()) / (den_map.max() - den_map.min() + 1e-8)
    heat = cv2.resize(den_norm, (frame_bgr.shape[1], frame_bgr.shape[0]))
    heatmap = (plt.cm.jet(heat)[:,:,:3] * 255).astype(np.uint8)
    overlay = (0.6 * heatmap + 0.4 * frame_bgr).astype(np.uint8)

    # draw vertical lines
    h, w = frame_bgr.shape[:2]
    if left_x is not None:
        cv2.line(overlay, (left_x, 0), (left_x, h), (0, 165, 255), 3)  # left line (exit) - orange
        cv2.putText(overlay, "EXIT", (max(5, left_x-50), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
    if right_x is not None:
        cv2.line(overlay, (right_x, 0), (right_x, h), (0,255,0), 3)  # right line (entry) - green
        cv2.putText(overlay, "ENTRY", (max(5, right_x-60), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # draw peaks
    if peaks:
        for (x,y) in peaks:
            cv2.circle(overlay, (int(x), int(y)), 3, (255,0,0), -1)

    # draw tracker IDs
    if tracker:
        for oid, centroid in tracker.objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.putText(overlay, str(oid), (cx+4, cy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # draw counts
    if entered is not None:
        cv2.putText(overlay, f"Entered: {entered}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    if left is not None:
        cv2.putText(overlay, f"Left: {left}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 2)

    cv2.imwrite(out_path, overlay)

# ------------------ Main processing ------------------ #
def process_stream(args):
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"Using device: {device}")
    model = load_weights_into_model(args.weights, device)
    print("Loaded model.")

    # tracker + counted flags
    tracker = CentroidTracker(max_lost=20, max_distance=80)
    tracker.counted = {}  # oid -> {"entered": False, "left": False}
    entered_count = 0
    left_count = 0

    # vertical lines configuration (fractions of width)
    LEFT_LINE_RATIO = 0.2   # left vertical line (exit)
    RIGHT_LINE_RATIO = 0.8  # right vertical line (entry)
    LINE_OFFSET = 12        # pixels tolerance

    # prepare input source
    if args.images:
        img_files = sorted([os.path.join(args.images, f) for f in os.listdir(args.images)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        total_frames = len(img_files)
        source_type = "images"
    else:
        src = args.video
        try:
            src_val = int(src)
        except Exception:
            src_val = src
        cap = cv2.VideoCapture(src_val)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else None
        source_type = "video"

    # output setup
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = args.out_csv or os.path.join(args.output_dir, "counts.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "frame_idx", "source", "raw_count", "ema", "median", "entered", "left"])

    # smoothing windows
    ema = None
    alpha = args.ema_alpha
    median_window = deque(maxlen=args.median_window)
    frame_idx = 0
    last_post = 0

    try:
        if source_type == "images":
            for frame_idx, img_path in enumerate(img_files, start=1):
                if args.frame_skip and (frame_idx % (args.frame_skip + 1) != 1):
                    continue

                bgr = cv2.imread(img_path)
                if bgr is None:
                    print("Failed to read", img_path); continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                t0 = time.time()
                raw_count, den_map = infer_count_and_map(model, rgb, device)

                # geometry for lines
                h, w = bgr.shape[:2]
                left_x = int(LEFT_LINE_RATIO * w)
                right_x = int(RIGHT_LINE_RATIO * w)

                # peaks & tracking
                peaks = extract_peaks_from_density(den_map, min_distance=8, threshold_rel=0.12, sigma=1.0)
                objects = tracker.update(peaks)

                # detect vertical-line crossings (x-coordinate)
                for oid in list(tracker.history.keys()):
                    hist = tracker.history.get(oid, [])
                    if len(hist) >= 2:
                        x_prev = hist[-2][0]
                        x_curr = hist[-1][0]
                        # ensure counted flags
                        if oid not in tracker.counted:
                            tracker.counted[oid] = {"entered": False, "left": False}
                        # left->right crossing of RIGHT line => ENTRY
                        if (x_prev < right_x - LINE_OFFSET) and (x_curr >= right_x + LINE_OFFSET) and (not tracker.counted[oid]["entered"]):
                            entered_count += 1
                            tracker.counted[oid]["entered"] = True
                        # right->left crossing of LEFT line => EXIT/LEFT
                        if (x_prev > left_x + LINE_OFFSET) and (x_curr <= left_x - LINE_OFFSET) and (not tracker.counted[oid]["left"]):
                            left_count += 1
                            tracker.counted[oid]["left"] = True

                # smoothing & logging
                if ema is None: ema = raw_count
                else: ema = alpha * raw_count + (1-alpha) * ema
                median_window.append(raw_count)
                median_val = float(np.median(np.array(median_window)))

                timestamp = datetime.utcnow().isoformat()
                csv_writer.writerow([timestamp, frame_idx, img_path, f"{raw_count:.4f}", f"{ema:.4f}", f"{median_val:.4f}", entered_count, left_count])
                csv_file.flush()

                # optional POST
                if args.post_url and (time.time() - last_post) > args.post_interval:
                    payload = {"camera_id": args.camera_id, "timestamp": timestamp, "frame": frame_idx,
                               "count": float(raw_count), "ema": float(ema), "median": float(median_val),
                               "entered": entered_count, "left": left_count}
                    try:
                        requests.post(args.post_url, json=payload, timeout=3)
                        last_post = time.time()
                    except Exception as e:
                        print("POST failed:", e)

                # save overlay
                if args.save_overlay and (frame_idx % args.save_overlay == 0):
                    outp = os.path.join(args.output_dir, f"overlay_{frame_idx:06d}.jpg")
                    save_overlay(bgr, den_map, outp, peaks=peaks, tracker=tracker, entered=entered_count, left=left_count, left_x=left_x, right_x=right_x)

                dt = time.time() - t0
                print(f"[{frame_idx}/{total_frames if total_frames else '?'}] raw={raw_count:.2f} ema={ema:.2f} median={median_val:.2f} entered={entered_count} left={left_count} dt={dt:.2f}s  src={img_path}")

                if args.target_fps > 0:
                    time.sleep(max(0, (1.0/args.target_fps) - dt))

        else:  # video
            start_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or read failed.")
                    break
                frame_idx += 1
                if args.frame_skip and (frame_idx % (args.frame_skip + 1) != 1):
                    continue

                orig_h, orig_w = frame.shape[:2]
                if args.resize_width and orig_w > args.resize_width:
                    scale = args.resize_width / float(orig_w)
                    frame = cv2.resize(frame, (args.resize_width, int(orig_h*scale)))
                bgr = frame
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                t0 = time.time()
                raw_count, den_map = infer_count_and_map(model, rgb, device)

                # geometry for lines
                h, w = bgr.shape[:2]
                left_x = int(LEFT_LINE_RATIO * w)
                right_x = int(RIGHT_LINE_RATIO * w)

                # peaks & tracking
                peaks = extract_peaks_from_density(den_map, min_distance=8, threshold_rel=0.12, sigma=1.0)
                objects = tracker.update(peaks)

                # detect vertical-line crossings (x-coordinate)
                for oid in list(tracker.history.keys()):
                    hist = tracker.history.get(oid, [])
                    if len(hist) >= 2:
                        x_prev = hist[-2][0]
                        x_curr = hist[-1][0]
                        if oid not in tracker.counted:
                            tracker.counted[oid] = {"entered": False, "left": False}
                        if (x_prev < right_x - LINE_OFFSET) and (x_curr >= right_x + LINE_OFFSET) and (not tracker.counted[oid]["entered"]):
                            entered_count += 1
                            tracker.counted[oid]["entered"] = True
                        if (x_prev > left_x + LINE_OFFSET) and (x_curr <= left_x - LINE_OFFSET) and (not tracker.counted[oid]["left"]):
                            left_count += 1
                            tracker.counted[oid]["left"] = True

                if ema is None: ema = raw_count
                else: ema = alpha * raw_count + (1-alpha) * ema
                median_window.append(raw_count)
                median_val = float(np.median(np.array(median_window)))

                timestamp = datetime.utcnow().isoformat()
                csv_writer.writerow([timestamp, frame_idx, args.video, f"{raw_count:.4f}", f"{ema:.4f}", f"{median_val:.4f}", entered_count, left_count])
                csv_file.flush()

                # optional POST
                if args.post_url and (time.time() - last_post) > args.post_interval:
                    payload = {"camera_id": args.camera_id, "timestamp": timestamp, "frame": frame_idx,
                               "count": float(raw_count), "ema": float(ema), "median": float(median_val),
                               "entered": entered_count, "left": left_count}
                    try:
                        requests.post(args.post_url, json=payload, timeout=3)
                        last_post = time.time()
                    except Exception as e:
                        print("POST failed:", e)

                # save overlay
                if args.save_overlay and (frame_idx % args.save_overlay == 0):
                    outp = os.path.join(args.output_dir, f"overlay_{frame_idx:06d}.jpg")
                    save_overlay(bgr, den_map, outp, peaks=peaks, tracker=tracker, entered=entered_count, left=left_count, left_x=left_x, right_x=right_x)

                dt = time.time() - t0
                print(f"[{frame_idx}] raw={raw_count:.2f} ema={ema:.2f} median={median_val:.2f} entered={entered_count} left={left_count} dt={dt:.2f}s")

                if args.target_fps > 0:
                    time.sleep(max(0.0, (1.0 / args.target_fps) - dt))

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        csv_file.close()
        if source_type == "video":
            cap.release()
        print("Done. CSV saved to:", csv_path)

# ------------------ CLI ------------------ #
def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to video file, or webcam index (0,1...) as int string")
    group.add_argument("--images", type=str, help="Path to a folder of images to process sequentially")
    p.add_argument("--weights", type=str, required=True, help="Path to weights.pth (Hugging Face downloaded file)")
    p.add_argument("--output-dir", type=str, default="outputs", help="Folder to save overlays and CSV")
    p.add_argument("--out-csv", type=str, default=None, help="CSV path (overrides default)")
    p.add_argument("--save-overlay", type=int, default=0, help="Save overlay every N frames (0 = disabled)")
    p.add_argument("--frame-skip", type=int, default=0, help="Skip processing every N frames (0 = process every frame)")
    p.add_argument("--target-fps", type=float, default=0.0, help="Target FPS for processing (0 = as fast as possible)")
    p.add_argument("--ema-alpha", type=float, default=0.35, help="EMA alpha (0..1) for smoothing")
    p.add_argument("--median-window", type=int, default=5, help="Window size for sliding median smoothing")
    p.add_argument("--post-url", type=str, default=None, help="Optional URL to POST counts as JSON")
    p.add_argument("--post-interval", type=float, default=2.0, help="Minimum seconds between POSTs")
    p.add_argument("--camera-id", type=str, default="camera_1", help="Camera ID to send in posts")
    p.add_argument("--resize-width", type=int, default=0, help="If >0, resize frame width to this (preserve aspect)")
    p.add_argument("--force-cpu", action="store_true", help="Force use CPU even if CUDA available")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.images and not os.path.isdir(args.images):
        raise SystemExit("Image folder not found: " + args.images)
    if not os.path.exists(args.weights):
        raise SystemExit("Weights file not found: " + args.weights)
    os.makedirs(args.output_dir, exist_ok=True)
    process_stream(args)
