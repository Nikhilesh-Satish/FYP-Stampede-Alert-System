import os
import time

import cv2
import numpy as np

from model_loader import load_model


class CameraWorker:
    def __init__(
        self,
        camera_id,
        video_path,
        shared_dict,
        shared_totals,
        shared_count_ts,
        shared_frames,
        running_status,
        lock,
    ):
        self.camera_id = camera_id
        self.video_path = video_path
        self.shared_dict = shared_dict
        self.shared_totals = shared_totals
        self.shared_count_ts = shared_count_ts
        self.shared_frames = shared_frames
        self.running_status = running_status
        self.lock = lock

        self.model = load_model()

        line_x_env = os.getenv("COUNT_LINE_X")
        self.LINE_X = int(line_x_env) if line_x_env is not None else None
        self.LINE_BAND = int(os.getenv("COUNT_LINE_BAND_PX", "12"))

        self.net_count = 0
        self.interval_in = 0
        self.interval_out = 0

        self.track_state = {}
        self.track_last_cross_ts = {}
        self.next_track_id = 1
        self.TRACK_TTL_SEC = float(os.getenv("TRACK_TTL_SEC", "1.5"))
        self.CROSSING_COOLDOWN_SEC = float(os.getenv("CROSSING_COOLDOWN_SEC", "1.0"))
        self.MIN_BOX_AREA = int(os.getenv("MIN_PERSON_BOX_AREA", "48"))
        self.TRACK_MATCH_IOU = float(os.getenv("TRACK_MATCH_IOU", "0.15"))
        self.TRACK_MATCH_CENTER = float(os.getenv("TRACK_MATCH_CENTER", "90"))

        self.det_conf = float(os.getenv("DETECTION_CONF", "0.10"))
        self.det_iou = float(os.getenv("DETECTION_IOU", "0.45"))
        self.imgsz = int(os.getenv("DETECTION_IMGSZ", "768"))
        self.max_det = int(os.getenv("DETECTION_MAX_DET", "1200"))
        self.jpeg_quality = int(os.getenv("STREAM_JPEG_QUALITY", "80"))
        self.stream_sleep_sec = float(os.getenv("STREAM_FRAME_DELAY_SEC", "0.03"))
        self.inference_every_n_frames = max(1, int(os.getenv("INFERENCE_EVERY_N_FRAMES", "3")))
        self.max_inference_fps = float(os.getenv("MAX_INFERENCE_FPS", "4"))
        self.inference_scale = float(os.getenv("INFERENCE_SCALE", "1.0"))
        self.realtime_sync = os.getenv("REALTIME_SYNC_ENABLED", "true").lower() != "false"
        self.max_frame_skip = max(0, int(os.getenv("MAX_FRAME_SKIP", "6")))
        self.source_fps_override = float(os.getenv("SOURCE_FPS_OVERRIDE", "0"))

        self.tile_enabled = os.getenv("TILE_INFERENCE_ENABLED", "true").lower() != "false"
        self.tile_size = int(os.getenv("TILE_SIZE", "640"))
        self.tile_overlap = float(os.getenv("TILE_OVERLAP", "0.35"))
        self.tile_batch = int(os.getenv("TILE_BATCH", "4"))

        self.last_reset_time = time.time()
        self.report_interval_sec = int(os.getenv("REPORT_INTERVAL_SEC", "10"))
        self.frame_index = 0
        self.last_inference_time = 0.0
        self.cached_boxes = []
        self.cached_ids = []
        self.source_frames_seen = 0
        self.playback_started_at = None
        self.source_frame_interval = None

    def _line_x(self):
        return self.LINE_X if self.LINE_X is not None else 320

    def _line_side(self, cx):
        line_x = self._line_x()
        if cx < line_x - self.LINE_BAND:
            return "left"
        if cx > line_x + self.LINE_BAND:
            return "right"
        return "middle"

    def _is_running(self):
        with self.lock:
            return self.running_status.get(self.camera_id, True)

    def reset_local_count(self):
        self.net_count = 0
        self.interval_in = 0
        self.interval_out = 0
        self.track_state.clear()
        self.track_last_cross_ts.clear()
        self.next_track_id = 1
        with self.lock:
            self.shared_dict[self.camera_id] = 0
            self.shared_totals[self.camera_id] = {"in_count": 0, "out_count": 0}
            self.shared_count_ts[self.camera_id] = int(time.time() * 1000)

    def _encode_jpeg(self, frame):
        ok, jpg = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return None
        return jpg.tobytes()

    def _box_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _center_distance(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

    def _nms(self, detections):
        if not detections:
            return []

        detections = sorted(detections, key=lambda det: det["score"], reverse=True)
        kept = []

        while detections:
            best = detections.pop(0)
            kept.append(best)
            detections = [
                det
                for det in detections
                if self._box_iou(best["box"], det["box"]) < self.det_iou
            ]

        return kept

    def _tile_positions(self, length):
        if length <= self.tile_size:
            return [0]

        stride = max(1, int(self.tile_size * (1.0 - self.tile_overlap)))
        positions = list(range(0, max(1, length - self.tile_size + 1), stride))
        if positions[-1] != length - self.tile_size:
            positions.append(length - self.tile_size)
        return positions

    def _make_tiles(self, frame):
        frame_h, frame_w = frame.shape[:2]
        if not self.tile_enabled:
            return [(frame, 0, 0)]

        tiles = []
        for tile_y in self._tile_positions(frame_h):
            for tile_x in self._tile_positions(frame_w):
                crop = frame[tile_y : tile_y + self.tile_size, tile_x : tile_x + self.tile_size]
                pad_h = self.tile_size - crop.shape[0]
                pad_w = self.tile_size - crop.shape[1]
                if pad_h > 0 or pad_w > 0:
                    crop = cv2.copyMakeBorder(
                        crop,
                        0,
                        pad_h,
                        0,
                        pad_w,
                        cv2.BORDER_CONSTANT,
                        value=(114, 114, 114),
                    )
                tiles.append((crop, tile_x, tile_y))
        return tiles

    def _prepare_inference_frame(self, frame):
        if self.inference_scale >= 0.999:
            return frame, 1.0

        scale = max(0.1, min(1.0, self.inference_scale))
        resized = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _predict_frame(self, frame):
        inference_frame, scale = self._prepare_inference_frame(frame)
        tiles = self._make_tiles(inference_frame)
        detections = []

        for start_idx in range(0, len(tiles), self.tile_batch):
            batch = tiles[start_idx : start_idx + self.tile_batch]
            crops = [crop for crop, _, _ in batch]
            results = self.model.predict(
                source=crops,
                classes=[0],
                conf=self.det_conf,
                iou=self.det_iou,
                imgsz=self.tile_size if self.tile_enabled else self.imgsz,
                max_det=self.max_det,
                verbose=False,
            )

            for result, (_, offset_x, offset_y) in zip(results, batch):
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = map(float, box)
                    x1 += offset_x
                    y1 += offset_y
                    x2 += offset_x
                    y2 += offset_y

                    if scale != 1.0:
                        inv_scale = 1.0 / scale
                        x1 *= inv_scale
                        y1 *= inv_scale
                        x2 *= inv_scale
                        y2 *= inv_scale

                    x1 = max(0.0, x1)
                    y1 = max(0.0, y1)
                    x2 = min(float(frame.shape[1]), x2)
                    y2 = min(float(frame.shape[0]), y2)
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    if area < self.MIN_BOX_AREA:
                        continue

                    detections.append(
                        {
                            "box": (x1, y1, x2, y2),
                            "score": float(score),
                        }
                    )

        return self._nms(detections)

    def _should_run_inference(self, now):
        if self.frame_index % self.inference_every_n_frames != 0:
            return False

        if self.max_inference_fps > 0:
            min_interval = 1.0 / self.max_inference_fps
            if now - self.last_inference_time < min_interval:
                return False

        return True

    def _resolve_source_frame_interval(self, cap):
        fps = self.source_fps_override
        if fps <= 0:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0

        if fps and np.isfinite(fps) and fps > 0.5:
            return 1.0 / fps

        return None

    def _pace_stream(self, cap):
        if not self.realtime_sync or self.source_frame_interval is None:
            if self.stream_sleep_sec > 0:
                time.sleep(self.stream_sleep_sec)
            return

        expected_elapsed = self.source_frames_seen * self.source_frame_interval
        actual_elapsed = time.time() - self.playback_started_at
        lag = actual_elapsed - expected_elapsed

        if lag > self.source_frame_interval and self.max_frame_skip > 0:
            frames_to_skip = min(self.max_frame_skip, int(lag / self.source_frame_interval))
            skipped = 0
            while skipped < frames_to_skip and cap.grab():
                skipped += 1

            if skipped:
                self.source_frames_seen += skipped
                self.frame_index += skipped
            return

        if lag < 0:
            time.sleep(min(-lag, self.stream_sleep_sec))
        elif self.stream_sleep_sec > 0:
            time.sleep(self.stream_sleep_sec)

    def _update_tracks(self, detections, now):
        active_track_ids = [
            track_id
            for track_id, state in self.track_state.items()
            if now - state["last_seen"] <= self.TRACK_TTL_SEC
        ]

        candidate_pairs = []
        for det_idx, detection in enumerate(detections):
            for track_id in active_track_ids:
                state = self.track_state[track_id]
                iou = self._box_iou(detection["box"], state["box"])
                center_distance = self._center_distance(detection["box"], state["box"])
                if iou >= self.TRACK_MATCH_IOU or center_distance <= self.TRACK_MATCH_CENTER:
                    candidate_pairs.append((iou - (center_distance / 10000.0), track_id, det_idx))

        candidate_pairs.sort(reverse=True)
        assigned_tracks = set()
        assigned_dets = set()
        matches = {}

        for _, track_id, det_idx in candidate_pairs:
            if track_id in assigned_tracks or det_idx in assigned_dets:
                continue
            assigned_tracks.add(track_id)
            assigned_dets.add(det_idx)
            matches[det_idx] = track_id

        boxes = []
        ids = []

        for det_idx, detection in enumerate(detections):
            box = detection["box"]
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            curr_side = self._line_side(cx)

            if det_idx in matches:
                track_id = matches[det_idx]
                state = self.track_state[track_id]
                state["box"] = box
                state["last_seen"] = now
                prev_cx = state["last_cx"]
                line_x = self._line_x()
                crossed_line = (
                    (prev_cx < line_x <= cx) or (prev_cx > line_x >= cx)
                )
                can_count = (
                    crossed_line
                    and (now - self.track_last_cross_ts.get(track_id, 0.0))
                    >= self.CROSSING_COOLDOWN_SEC
                )
                if can_count:
                    if prev_cx > cx:
                        self.interval_out += 1
                        self.net_count -= 1
                    elif prev_cx < cx:
                        self.interval_in += 1
                        self.net_count += 1

                    with self.lock:
                        self.shared_dict[self.camera_id] = self.net_count
                        self.shared_totals[self.camera_id] = {
                            "in_count": self.interval_in,
                            "out_count": self.interval_out,
                        }
                        self.shared_count_ts[self.camera_id] = int(time.time() * 1000)
                    self.track_last_cross_ts[track_id] = now

                state["last_cx"] = cx
                state["last_side"] = curr_side
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.track_state[track_id] = {
                    "box": box,
                    "last_seen": now,
                    "last_side": curr_side,
                    "last_cx": cx,
                }

            boxes.append((x1, y1, x2, y2))
            ids.append(track_id)

        stale_ids = [
            track_id
            for track_id, state in self.track_state.items()
            if now - state["last_seen"] > self.TRACK_TTL_SEC
        ]
        for track_id in stale_ids:
            self.track_state.pop(track_id, None)
            self.track_last_cross_ts.pop(track_id, None)

        return boxes, ids

    def _draw_overlay(self, frame, boxes, ids):
        overlay = frame.copy()
        line_x = self.LINE_X if self.LINE_X is not None else overlay.shape[1] // 2
        cv2.line(
            overlay,
            (line_x, 0),
            (line_x, overlay.shape[0]),
            (0, 255, 255),
            2,
        )
        cv2.rectangle(
            overlay,
            (line_x - self.LINE_BAND, 0),
            (line_x + self.LINE_BAND, overlay.shape[0]),
            (255, 255, 0),
            1,
        )

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            if max(0, x2 - x1) * max(0, y2 - y1) < self.MIN_BOX_AREA:
                continue
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                overlay,
                f"ID {track_id}",
                (x1, max(14, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            overlay,
            f"IN:{self.interval_in} OUT:{self.interval_out}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            "Tiled YOLOv8 + tracker | left->right = IN, right->left = OUT",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return overlay

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Camera {self.camera_id}: failed to open '{self.video_path}'")
            return

        self.source_frame_interval = self._resolve_source_frame_interval(cap)
        self.playback_started_at = time.time()

        print(f"Camera {self.camera_id} started processing")

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.source_frames_seen = 0
                self.playback_started_at = time.time()
                continue
            self.frame_index += 1
            self.source_frames_seen += 1
            if self.LINE_X is None:
                self.LINE_X = frame.shape[1] // 2

            if not self._is_running():
                overlay = frame.copy()
                cv2.putText(
                    overlay,
                    "Monitoring paused",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2,
                )
                raw_jpg = self._encode_jpeg(frame)
                overlay_jpg = self._encode_jpeg(overlay)
                if raw_jpg and overlay_jpg:
                    with self.lock:
                        self.shared_frames[self.camera_id] = {
                            "raw": raw_jpg,
                            "overlay": overlay_jpg,
                            "timestamp": int(time.time() * 1000),
                        }
                self._pace_stream(cap)
                continue

            now = time.time()
            if self._should_run_inference(now):
                detections = self._predict_frame(frame)
                self.cached_boxes, self.cached_ids = self._update_tracks(detections, now)
                self.last_inference_time = now

            raw_jpg = self._encode_jpeg(frame)
            overlay_jpg = self._encode_jpeg(self._draw_overlay(frame, self.cached_boxes, self.cached_ids))
            if raw_jpg and overlay_jpg:
                with self.lock:
                    self.shared_frames[self.camera_id] = {
                        "raw": raw_jpg,
                        "overlay": overlay_jpg,
                        "timestamp": int(time.time() * 1000),
                    }

            if time.time() - self.last_reset_time >= self.report_interval_sec:
                with self.lock:
                    self.shared_dict[self.camera_id] = self.net_count
                    self.shared_totals[self.camera_id] = {
                        "in_count": self.interval_in,
                        "out_count": self.interval_out,
                    }
                    self.shared_count_ts[self.camera_id] = int(time.time() * 1000)

                print(
                    f"{self.camera_id} report | IN:{self.interval_in} "
                    f"OUT:{self.interval_out} NET:{self.net_count}"
                )
                self.last_reset_time = time.time()

            self._pace_stream(cap)
