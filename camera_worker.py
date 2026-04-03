import os
import threading
import time

import cv2
import numpy as np
import yaml

from model_loader import load_model
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace


class TrackerResults:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    @property
    def xywh(self):
        if len(self.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        xywh = self.xyxy.copy()
        xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
        xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
        xywh[:, 0] = xywh[:, 0] + xywh[:, 2] / 2
        xywh[:, 1] = xywh[:, 1] + xywh[:, 3] / 2
        return xywh

    def __len__(self):
        return len(self.xyxy)

    def __getitem__(self, idx):
        return TrackerResults(self.xyxy[idx], self.conf[idx], self.cls[idx])


def _to_numpy(array_like):
    cpu_fn = getattr(array_like, "cpu", None)
    if callable(cpu_fn):
        array_like = cpu_fn()
    numpy_fn = getattr(array_like, "numpy", None)
    if callable(numpy_fn):
        return numpy_fn()
    return np.asarray(array_like)


class CameraWorker:
    def __init__(
        self,
        camera_id,
        video_path,
        count_axis,
        in_direction,
        shot_type,
        processing_preset,
        counting_enabled,
        density_enabled,
        monitored_area_sqm,
        shared_dict,
        shared_totals,
        shared_count_ts,
        shared_frames,
        shared_density_stats,
        running_status,
        lock,
    ):
        self.camera_id = camera_id
        self.video_path = video_path
        self.shared_dict = shared_dict
        self.shared_totals = shared_totals
        self.shared_count_ts = shared_count_ts
        self.shared_frames = shared_frames
        self.shared_density_stats = shared_density_stats
        self.running_status = running_status
        self.lock = lock

        self.model = load_model()

        self.count_axis = count_axis if count_axis in {"x", "y"} else self._resolve_count_axis()
        self.in_direction = in_direction if in_direction in {"positive", "negative"} else "positive"
        self.shot_type = shot_type if shot_type in {"drone", "ground"} else "ground"
        self.processing_preset = (
            processing_preset if processing_preset in {"fast", "balanced", "accurate"} else "balanced"
        )
        self.counting_enabled = bool(counting_enabled)
        self.density_enabled = bool(density_enabled)
        self.monitored_area_sqm = float(monitored_area_sqm) if monitored_area_sqm else None
        line_x_env = os.getenv("COUNT_LINE_X")
        line_y_env = os.getenv("COUNT_LINE_Y")
        self.LINE_X = int(line_x_env) if line_x_env is not None else None
        self.LINE_Y = int(line_y_env) if line_y_env is not None else None
        self.LINE_BAND = int(os.getenv("COUNT_LINE_BAND_PX", "20"))
        self.OUTER_BAND = int(os.getenv("COUNT_OUTER_BAND_PX", "120"))
        self.ground_anchor_ratio = float(os.getenv("GROUND_COUNT_ANCHOR_RATIO", "0.85"))
        self.perspective_enabled = os.getenv("PERSPECTIVE_COUNTING_ENABLED", "true").lower() != "false"
        self.perspective_far_scale = float(os.getenv("PERSPECTIVE_FAR_SCALE", "0.45"))
        self.perspective_far_y_ratio = float(os.getenv("PERSPECTIVE_FAR_Y_RATIO", "0.18"))

        self.net_count = 0
        self.interval_in = 0
        self.interval_out = 0

        self.track_state = {}
        self.track_last_cross_ts = {}
        self.next_track_id = 1
        self.TRACK_TTL_SEC = float(os.getenv("TRACK_TTL_SEC", "2.5"))
        self.TRACK_MEMORY_TIMEOUT_SEC = float(os.getenv("TRACK_MEMORY_TIMEOUT_SEC", "6.0"))
        self.LOST_TRACK_REID_TIMEOUT_SEC = float(os.getenv("LOST_TRACK_REID_TIMEOUT_SEC", "2.0"))
        self.OVERLAY_TRACK_GRACE_SEC = float(os.getenv("OVERLAY_TRACK_GRACE_SEC", "0.6"))
        self.CROSSING_COOLDOWN_SEC = float(os.getenv("CROSSING_COOLDOWN_SEC", "1.0"))
        self.MIN_TRACK_AGE_FRAMES = int(os.getenv("MIN_TRACK_AGE_FRAMES", "2"))
        self.MIN_BOX_AREA = int(os.getenv("MIN_PERSON_BOX_AREA", "48"))
        self.TRACK_MATCH_IOU = float(os.getenv("TRACK_MATCH_IOU", "0.15"))
        self.TRACK_MATCH_CENTER = float(os.getenv("TRACK_MATCH_CENTER", "110"))

        self.det_conf = float(os.getenv("DETECTION_CONF", "0.06"))
        self.drone_det_conf = float(os.getenv("DRONE_DETECTION_CONF", "0.04"))
        self.ground_det_conf = float(os.getenv("GROUND_DETECTION_CONF", "0.05"))
        self.det_iou = float(os.getenv("DETECTION_IOU", "0.45"))
        self.imgsz = int(os.getenv("DETECTION_IMGSZ", "704"))
        self.max_det = int(os.getenv("DETECTION_MAX_DET", "1500"))
        self.jpeg_quality = int(os.getenv("STREAM_JPEG_QUALITY", "78"))
        self.stream_scale = float(os.getenv("STREAM_SCALE", "0.75"))
        self.stream_publish_fps = float(os.getenv("STREAM_PUBLISH_FPS", "10"))
        self.stream_sleep_sec = float(os.getenv("STREAM_FRAME_DELAY_SEC", "0.03"))
        self.inference_every_n_frames = max(1, int(os.getenv("INFERENCE_EVERY_N_FRAMES", "2")))
        self.max_inference_fps = float(os.getenv("MAX_INFERENCE_FPS", "6"))
        self.inference_scale = float(os.getenv("INFERENCE_SCALE", "1.0"))
        self.realtime_sync = os.getenv("REALTIME_SYNC_ENABLED", "true").lower() != "false"
        self.max_frame_skip = max(0, int(os.getenv("MAX_FRAME_SKIP", "0")))
        self.source_fps_override = float(os.getenv("SOURCE_FPS_OVERRIDE", "0"))

        self.tile_enabled = os.getenv("TILE_INFERENCE_ENABLED", "true").lower() != "false"
        self.tile_size = int(os.getenv("TILE_SIZE", "512"))
        self.tile_overlap = float(os.getenv("TILE_OVERLAP", "0.30"))
        self.tile_batch = int(os.getenv("TILE_BATCH", "6"))
        self.ground_split_overlap = float(os.getenv("GROUND_SPLIT_OVERLAP", "0.16"))
        self.ground_horizontal_splits = max(1, int(os.getenv("GROUND_HORIZONTAL_SPLITS", "3")))
        self.drone_tile_size = int(os.getenv("DRONE_TILE_SIZE", "512"))
        self.drone_tile_overlap = float(os.getenv("DRONE_TILE_OVERLAP", "0.35"))
        self._apply_processing_preset()

        self.last_reset_time = time.time()
        self.report_interval_sec = int(os.getenv("REPORT_INTERVAL_SEC", "10"))
        self.frame_index = 0
        self.last_inference_time = 0.0
        self.cached_boxes = []
        self.cached_ids = []
        self.cached_display_boxes = []
        self.cached_display_ids = []
        self.cached_overlay_jpg = None
        self.display_frames_seen = 0
        self.display_started_at = None
        self.display_frame_interval = None
        self.process_frames_seen = 0
        self.process_started_at = None
        self.process_frame_interval = None
        self.last_publish_time = 0.0
        self.density_ema_alpha = float(os.getenv("DENSITY_EMA_ALPHA", "0.35"))
        self.density_update_interval_sec = float(os.getenv("DENSITY_UPDATE_INTERVAL_SEC", "10.0"))
        self.last_density_update_time = 0.0
        self.current_weighted_people = 0.0
        self.current_density_score = 0.0
        self.smoothed_density_score = 0.0
        self.state_lock = threading.Lock()
        self.process_failed = False
        self.byte_tracker = None
        self.tracker_args = self._load_tracker_args()
        self.lost_track_memory = {}

    def _apply_processing_preset(self):
        if self.processing_preset == "fast":
            self.imgsz = 576
            self.max_det = 900
            self.stream_scale = 0.65
            self.stream_publish_fps = 8.0
            self.inference_every_n_frames = max(self.inference_every_n_frames, 3)
            self.max_inference_fps = 4.0 if self.max_inference_fps <= 0 else min(self.max_inference_fps, 4.0)
            self.inference_scale = min(self.inference_scale, 0.85)
            self.tile_batch = max(self.tile_batch, 8)
            self.ground_horizontal_splits = min(self.ground_horizontal_splits, 2)
            self.ground_split_overlap = min(self.ground_split_overlap, 0.12)
            self.drone_tile_size = max(self.drone_tile_size, 576)
            self.drone_tile_overlap = min(self.drone_tile_overlap, 0.25)
            return

        if self.processing_preset == "accurate":
            self.imgsz = max(self.imgsz, 832)
            self.max_det = max(self.max_det, 2200)
            self.stream_scale = max(self.stream_scale, 0.8)
            self.stream_publish_fps = max(self.stream_publish_fps, 10.0)
            self.inference_every_n_frames = 1
            self.max_inference_fps = 0.0
            self.inference_scale = 1.0
            self.tile_batch = min(self.tile_batch, 4)
            self.ground_horizontal_splits = max(self.ground_horizontal_splits, 3)
            self.ground_split_overlap = max(self.ground_split_overlap, 0.2)
            self.drone_tile_size = min(self.drone_tile_size, 448)
            self.drone_tile_overlap = max(self.drone_tile_overlap, 0.4)

    def _resolve_count_axis(self):
        explicit_axis = os.getenv("COUNT_AXIS", "").strip().lower()
        if explicit_axis in {"x", "y"}:
            return explicit_axis

        return "x"

    def _reset_tracking_state(self):
        self.track_state.clear()
        self.track_last_cross_ts.clear()
        self.next_track_id = 1
        self.cached_boxes = []
        self.cached_ids = []
        self.cached_display_boxes = []
        self.cached_display_ids = []
        self.cached_overlay_jpg = None
        self.last_inference_time = 0.0
        self.lost_track_memory.clear()
        self.last_density_update_time = 0.0
        self.current_weighted_people = 0.0
        self.current_density_score = 0.0
        self.smoothed_density_score = 0.0
        if self.byte_tracker is not None:
            self.byte_tracker.reset()

    def _density_band_weights(self, frame_h):
        if self.shot_type == "drone":
            return [1.0, 1.0, 1.0]

        band_count = 3
        centers = [int(((idx + 0.5) * frame_h) / band_count) for idx in range(band_count)]
        raw = []
        for cy in centers:
            scale = max(0.2, self._perspective_scale(cy, frame_h))
            raw.append(1.0 / scale)
        near_weight = raw[-1] if raw else 1.0
        return [max(1.0, value / max(near_weight, 1e-6)) for value in raw]

    def _density_band_index(self, cy, frame_h):
        frame_h = max(int(frame_h), 1)
        normalized_y = max(0, min(frame_h - 1, int(cy)))
        return min(2, int((normalized_y / frame_h) * 3))

    def _update_density_stats(self, tracked_detections, frame_shape, now):
        if (
            self.last_density_update_time > 0
            and self.density_update_interval_sec > 0
            and (now - self.last_density_update_time) < self.density_update_interval_sec
        ):
            return

        if not self.density_enabled or not self.monitored_area_sqm or self.monitored_area_sqm <= 0:
            self.current_weighted_people = 0.0
            self.current_density_score = 0.0
            self.smoothed_density_score = 0.0
        else:
            frame_h = frame_shape[0]
            weights = self._density_band_weights(frame_h)
            weighted_people = 0.0
            for det in tracked_detections:
                x1, y1, x2, y2 = det["box"]
                anchor_x, anchor_y = self._count_anchor((x1, y1, x2, y2))
                band_idx = self._density_band_index(anchor_y, frame_h)
                weighted_people += weights[band_idx]

            density_score = weighted_people / self.monitored_area_sqm
            if self.smoothed_density_score <= 0:
                smoothed = density_score
            else:
                alpha = max(0.05, min(1.0, self.density_ema_alpha))
                smoothed = (alpha * density_score) + ((1.0 - alpha) * self.smoothed_density_score)

            self.current_weighted_people = weighted_people
            self.current_density_score = density_score
            self.smoothed_density_score = smoothed

        self.last_density_update_time = now

        with self.lock:
            self.shared_density_stats[self.camera_id] = {
                "density_enabled": self.density_enabled,
                "weighted_people": round(self.current_weighted_people, 3),
                "density_score": round(self.current_density_score, 4),
                "smoothed_density_score": round(self.smoothed_density_score, 4),
                "monitored_area_sqm": self.monitored_area_sqm,
                "updated_at": int(now * 1000),
            }

    def _line_position(self):
        if self.count_axis == "y":
            return self.LINE_Y if self.LINE_Y is not None else 180
        return self.LINE_X if self.LINE_X is not None else 320

    def _use_perspective_counting(self):
        return self.perspective_enabled and self.shot_type == "ground" and self.count_axis == "x"

    def _perspective_scale(self, cy, frame_height=None):
        if not self._use_perspective_counting():
            return 1.0

        if frame_height is None:
            frame_height = max((self.LINE_Y or 0) * 2, 360)
        frame_height = max(int(frame_height), 2)
        far_y = int(frame_height * self.perspective_far_y_ratio)
        near_y = frame_height - 1
        if near_y <= far_y:
            return 1.0

        progress = (cy - far_y) / float(near_y - far_y)
        progress = max(0.0, min(1.0, progress))
        far_scale = max(0.15, min(1.0, self.perspective_far_scale))
        return far_scale + ((1.0 - far_scale) * progress)

    def _dynamic_line_band(self, cy, frame_height=None):
        scale = self._perspective_scale(cy, frame_height)
        return max(4, int(round(self.LINE_BAND * scale)))

    def _dynamic_outer_band(self, cy, frame_height=None):
        scale = self._perspective_scale(cy, frame_height)
        return max(self._dynamic_line_band(cy, frame_height) + 8, int(round(self.OUTER_BAND * scale)))

    def _line_side(self, cx, cy):
        value = cy if self.count_axis == "y" else cx
        line_position = self._line_position()
        line_band = self._dynamic_line_band(cy)
        if value < line_position - line_band:
            return "left"
        if value > line_position + line_band:
            return "right"
        return "middle"

    def _corridor_zone(self, cx, cy):
        value = cy if self.count_axis == "y" else cx
        line_position = self._line_position()
        line_band = self._dynamic_line_band(cy)
        outer_band = self._dynamic_outer_band(cy)
        if value < line_position - outer_band:
            return "outside_left"
        if value < line_position - line_band:
            return "left"
        if value <= line_position + line_band:
            return "middle"
        if value <= line_position + outer_band:
            return "right"
        return "outside_right"

    def _box_side(self, box):
        if self.shot_type == "drone":
            x1, y1, x2, y2 = box
            line_position = self._line_position()
            if self.count_axis == "y":
                if y2 < line_position - self.LINE_BAND:
                    return "left"
                if y1 > line_position + self.LINE_BAND:
                    return "right"
                return "middle"

            if x2 < line_position - self.LINE_BAND:
                return "left"
            if x1 > line_position + self.LINE_BAND:
                return "right"
            return "middle"

        anchor_x, anchor_y = self._count_anchor(box)
        return self._line_side(anchor_x, anchor_y)

    def _box_corridor_zone(self, box):
        if self.shot_type == "drone":
            x1, y1, x2, y2 = box
            line_position = self._line_position()
            if self.count_axis == "y":
                if y2 < line_position - self.OUTER_BAND:
                    return "outside_left"
                if y2 < line_position - self.LINE_BAND:
                    return "left"
                if y1 <= line_position + self.LINE_BAND:
                    return "middle"
                if y1 <= line_position + self.OUTER_BAND:
                    return "right"
                return "outside_right"

            if x2 < line_position - self.OUTER_BAND:
                return "outside_left"
            if x2 < line_position - self.LINE_BAND:
                return "left"
            if x1 <= line_position + self.LINE_BAND:
                return "middle"
            if x1 <= line_position + self.OUTER_BAND:
                return "right"
            return "outside_right"

        anchor_x, anchor_y = self._count_anchor(box)
        return self._corridor_zone(anchor_x, anchor_y)

    def _count_anchor(self, box):
        x1, y1, x2, y2 = box
        anchor_x = int((x1 + x2) / 2)
        if self.shot_type == "drone":
            anchor_y = int((y1 + y2) / 2)
        else:
            box_height = max(1, y2 - y1)
            anchor_y = int(y1 + (box_height * self.ground_anchor_ratio))
        return anchor_x, anchor_y

    def _remember_lost_track(self, track_id, state):
        self.lost_track_memory[track_id] = {
            "box": state["box"],
            "last_seen": state["last_seen"],
            "last_side": state["last_side"],
            "last_non_middle_side": state.get("last_non_middle_side"),
            "entered_band_from": state.get("entered_band_from"),
            "age_frames": state.get("age_frames", 1),
            "last_anchor_x": state.get("last_anchor_x"),
            "last_anchor_y": state.get("last_anchor_y"),
        }

    def _prune_lost_track_memory(self, now):
        stale = [
            track_id
            for track_id, state in self.lost_track_memory.items()
            if now - state["last_seen"] > self.LOST_TRACK_REID_TIMEOUT_SEC
        ]
        for track_id in stale:
            self.lost_track_memory.pop(track_id, None)

    def _recover_track_id(self, box, curr_side, now):
        self._prune_lost_track_memory(now)
        best_track_id = None
        best_score = float("inf")
        for candidate_id, state in self.lost_track_memory.items():
            candidate_side = state.get("last_non_middle_side") or state.get("last_side")
            if candidate_side not in {curr_side, "middle", None} and curr_side != "middle":
                continue
            iou = self._box_iou(box, state["box"])
            center_distance = self._center_distance(box, state["box"])
            if iou < self.TRACK_MATCH_IOU and center_distance > self.TRACK_MATCH_CENTER:
                continue
            score = (1.0 - iou) + (center_distance / 1000.0)
            if score < best_score:
                best_score = score
                best_track_id = candidate_id

        if best_track_id is None:
            return None

        recovered = self.lost_track_memory.pop(best_track_id)
        recovered.update(
            {
                "box": box,
                "last_seen": now,
                "last_side": curr_side,
            }
        )
        self.track_state[best_track_id] = recovered
        return best_track_id

    def _load_tracker_args(self):
        with open("bytetrack_crowd.yaml", "r", encoding="utf-8") as handle:
            return IterableSimpleNamespace(**yaml.safe_load(handle))

    def _get_tracker(self):
        if self.byte_tracker is None:
            frame_rate = int(round(1.0 / self.process_frame_interval)) if self.process_frame_interval else 30
            self.byte_tracker = BYTETracker(self.tracker_args, frame_rate=max(frame_rate, 1))
        return self.byte_tracker

    def _direction_hint(self):
        if self.count_axis == "y":
            return (
                "top->bottom = IN, bottom->top = OUT"
                if self.in_direction == "positive"
                else "bottom->top = IN, top->bottom = OUT"
            )
        return (
            "left->right = IN, right->left = OUT"
            if self.in_direction == "positive"
            else "right->left = IN, left->right = OUT"
        )

    def _is_running(self):
        with self.lock:
            return self.running_status.get(self.camera_id, True)

    def reset_local_count(self):
        self.net_count = 0
        self.interval_in = 0
        self.interval_out = 0
        self._reset_tracking_state()
        with self.lock:
            self.shared_dict[self.camera_id] = 0
            self.shared_totals[self.camera_id] = {"in_count": 0, "out_count": 0}
            self.shared_count_ts[self.camera_id] = int(time.time() * 1000)
            self.shared_density_stats[self.camera_id] = {
                "density_enabled": self.density_enabled,
                "weighted_people": 0.0,
                "density_score": 0.0,
                "smoothed_density_score": 0.0,
                "monitored_area_sqm": self.monitored_area_sqm,
                "updated_at": int(time.time() * 1000),
            }

    def _encode_jpeg(self, frame):
        if self.stream_scale < 0.999:
            scale = max(0.2, min(1.0, self.stream_scale))
            frame = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
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
        tile_size = self.drone_tile_size if self.shot_type == "drone" else self.tile_size
        tile_overlap = self.drone_tile_overlap if self.shot_type == "drone" else self.tile_overlap
        if length <= tile_size:
            return [0]

        stride = max(1, int(tile_size * (1.0 - tile_overlap)))
        positions = list(range(0, max(1, length - tile_size + 1), stride))
        if positions[-1] != length - tile_size:
            positions.append(length - tile_size)
        return positions

    def _ground_band_slices(self, frame_h):
        split_count = max(1, self.ground_horizontal_splits)
        base_height = max(1, int(np.ceil(frame_h / float(split_count))))
        overlap_px = int(base_height * max(0.0, min(0.5, self.ground_split_overlap)))
        slices = []

        for idx in range(split_count):
            start_y = idx * base_height
            end_y = min(frame_h, (idx + 1) * base_height)
            if idx > 0:
                start_y = max(0, start_y - overlap_px)
            if idx < split_count - 1:
                end_y = min(frame_h, end_y + overlap_px)
            if end_y <= start_y:
                continue
            slices.append((start_y, end_y))

        return slices or [(0, frame_h)]

    def _make_tiles(self, frame):
        frame_h, frame_w = frame.shape[:2]
        if self.shot_type == "ground":
            tiles = []
            # Split ground-view footage by depth bands so far-away people are enlarged
            # without paying the cost of a dense full-frame tile grid.
            for start_y, end_y in self._ground_band_slices(frame_h):
                crop = frame[start_y:end_y, 0:frame_w]
                tiles.append((crop, 0, start_y))
            return tiles

        if not self.tile_enabled:
            return [(frame, 0, 0)]

        tile_size = self.drone_tile_size if self.shot_type == "drone" else self.tile_size
        tiles = []
        for tile_y in self._tile_positions(frame_h):
            for tile_x in self._tile_positions(frame_w):
                crop = frame[tile_y : tile_y + tile_size, tile_x : tile_x + tile_size]
                pad_h = tile_size - crop.shape[0]
                pad_w = tile_size - crop.shape[1]
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
        conf = self.drone_det_conf if self.shot_type == "drone" else self.ground_det_conf
        imgsz = self.drone_tile_size if (self.tile_enabled and self.shot_type == "drone") else self.imgsz

        for start_idx in range(0, len(tiles), self.tile_batch):
            batch = tiles[start_idx : start_idx + self.tile_batch]
            crops = [crop for crop, _, _ in batch]
            results = self.model.predict(
                source=crops,
                classes=[0],
                conf=conf,
                iou=self.det_iou,
                imgsz=imgsz,
                max_det=self.max_det,
                verbose=False,
            )

            for result, (_, offset_x, offset_y) in zip(results, batch):
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes = np.asarray(_to_numpy(result.boxes.xyxy), dtype=np.float32).tolist()
                scores = np.asarray(_to_numpy(result.boxes.conf), dtype=np.float32).tolist()

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

    def _track_detections(self, detections, frame):
        if not detections:
            self._get_tracker().update(
                TrackerResults(np.empty((0, 4)), np.empty((0,)), np.empty((0,))),
                img=frame,
            )
            return []

        xyxy = np.asarray([det["box"] for det in detections], dtype=np.float32)
        conf = np.asarray([det["score"] for det in detections], dtype=np.float32)
        cls = np.zeros(len(detections), dtype=np.float32)
        tracked = self._get_tracker().update(TrackerResults(xyxy, conf, cls), img=frame)

        tracked_detections = []
        for row in tracked:
            x1, y1, x2, y2, track_id, score, cls_id, _ = row.tolist()
            tracked_detections.append(
                {
                    "box": (x1, y1, x2, y2),
                    "score": score,
                    "track_id": int(track_id),
                    "cls": int(cls_id),
                }
            )
        return tracked_detections

    def _should_run_inference(self, now):
        if self.frame_index % self.inference_every_n_frames != 0:
            return False

        if self.max_inference_fps > 0:
            min_interval = 1.0 / self.max_inference_fps
            if now - self.last_inference_time < min_interval:
                return False

        return True

    def _resolve_frame_interval(self, cap):
        fps = self.source_fps_override
        if fps <= 0:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0

        if fps and np.isfinite(fps) and fps > 0.5:
            return 1.0 / fps

        return None

    def _pace_capture(self, cap, frame_interval, frames_seen, started_at):
        if not self.realtime_sync or frame_interval is None:
            if self.stream_sleep_sec > 0:
                time.sleep(self.stream_sleep_sec)
            return

        expected_elapsed = frames_seen * frame_interval
        actual_elapsed = time.time() - started_at
        lag = actual_elapsed - expected_elapsed

        if lag > frame_interval and self.max_frame_skip > 0:
            frames_to_skip = min(self.max_frame_skip, int(lag / frame_interval))
            skipped = 0
            while skipped < frames_to_skip and cap.grab():
                skipped += 1

            return skipped

        if lag < 0:
            time.sleep(min(-lag, self.stream_sleep_sec))
        elif self.stream_sleep_sec > 0:
            time.sleep(self.stream_sleep_sec)
        return 0

    def _restart_video_loop(self, cap):
        self._reset_tracking_state()

        if cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            return cap

        cap.release()
        restarted_cap = cv2.VideoCapture(self.video_path)
        if restarted_cap.isOpened():
            return restarted_cap

        print(f"Camera {self.camera_id}: failed to restart '{self.video_path}', retrying...")
        time.sleep(0.5)
        return restarted_cap

    def _publish_frame(self, frame, is_running, overlay_override=None):
        if self.stream_publish_fps > 0:
            min_interval = 1.0 / self.stream_publish_fps
            now = time.time()
            if now - self.last_publish_time < min_interval:
                return
            self.last_publish_time = now

        raw_jpg = self._encode_jpeg(frame)
        if not raw_jpg:
            return

        if is_running:
            overlay_jpg = overlay_override
            if overlay_jpg is None:
                with self.state_lock:
                    overlay_jpg = self.cached_overlay_jpg
        else:
            overlay_frame = frame.copy()
            cv2.putText(
                overlay_frame,
                "Monitoring paused",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )
            overlay_jpg = self._encode_jpeg(overlay_frame)
            if not overlay_jpg:
                return

        if is_running and not overlay_jpg:
            return

        with self.lock:
            self.shared_frames[self.camera_id] = {
                "raw": raw_jpg,
                "overlay": overlay_jpg,
                "timestamp": int(time.time() * 1000),
            }

    def _processing_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Camera {self.camera_id}: failed to open '{self.video_path}' for processing")
            self.process_failed = True
            return

        self.process_frame_interval = self._resolve_frame_interval(cap)
        self.process_started_at = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                cap = self._restart_video_loop(cap)
                if not cap or not cap.isOpened():
                    self.process_failed = True
                    return
                self.process_frame_interval = self._resolve_frame_interval(cap)
                self.process_frames_seen = 0
                self.process_started_at = time.time()
                continue

            self.frame_index += 1
            self.process_frames_seen += 1
            if self.count_axis == "y" and self.LINE_Y is None:
                self.LINE_Y = frame.shape[0] // 2
            if self.count_axis == "x" and self.LINE_X is None:
                self.LINE_X = frame.shape[1] // 2

            if not self._is_running():
                self._publish_frame(frame, False)
                time.sleep(0.02)
                skipped = self._pace_capture(
                    cap,
                    self.process_frame_interval,
                    self.process_frames_seen,
                    self.process_started_at,
                )
                if skipped:
                    self.process_frames_seen += skipped
                continue

            now = time.time()
            if self._should_run_inference(now):
                detections = self._predict_frame(frame)
                tracked_detections = self._track_detections(detections, frame)
                self._update_density_stats(tracked_detections, frame.shape, now)
                if self.counting_enabled:
                    boxes, ids = self._update_tracks(tracked_detections, now)
                else:
                    boxes = [tuple(map(int, det["box"])) for det in tracked_detections]
                    ids = [det.get("track_id") for det in tracked_detections]
                overlay_jpg = self._encode_jpeg(
                    self._draw_overlay(frame, boxes, ids)
                )
                with self.state_lock:
                    self.cached_boxes = boxes
                    self.cached_ids = ids
                    self.cached_display_boxes = boxes
                    self.cached_display_ids = ids
                    self.cached_overlay_jpg = overlay_jpg
                self._publish_frame(frame, True, overlay_override=overlay_jpg)
                self.last_inference_time = now
            else:
                self._publish_frame(frame, True)

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

            skipped = self._pace_capture(
                cap,
                self.process_frame_interval,
                self.process_frames_seen,
                self.process_started_at,
            )
            if skipped:
                self.process_frames_seen += skipped

    def _update_tracks(self, detections, now):
        if not self.counting_enabled:
            return [], []

        for detection in detections:
            box = detection["box"]
            x1, y1, x2, y2 = map(int, box)
            anchor_x, anchor_y = self._count_anchor((x1, y1, x2, y2))
            curr_side = self._box_side((x1, y1, x2, y2))
            curr_zone = self._box_corridor_zone((x1, y1, x2, y2))
            track_id = detection.get("track_id")

            if curr_zone.startswith("outside_"):
                if track_id in self.track_state:
                    state = self.track_state.pop(track_id, None)
                    if state is not None:
                        self._remember_lost_track(track_id, state)
                    self.track_last_cross_ts.pop(track_id, None)
                continue

            if track_id not in self.track_state:
                recovered_track_id = self._recover_track_id(box, curr_side, now)
                if recovered_track_id is not None:
                    track_id = recovered_track_id

            if track_id in self.track_state:
                state = self.track_state[track_id]
                state["box"] = box
                state["last_seen"] = now
                state["age_frames"] += 1
                previous_side = state.get("last_non_middle_side") or state["last_side"]
                crossed_line = (
                    state["age_frames"] >= self.MIN_TRACK_AGE_FRAMES
                    and previous_side in {"left", "right"}
                    and curr_side in {"middle", "left", "right"}
                    and curr_side != previous_side
                )
                can_count = (
                    crossed_line
                    and (now - self.track_last_cross_ts.get(track_id, 0.0))
                    >= self.CROSSING_COOLDOWN_SEC
                )
                if can_count:
                    motion_direction = (
                        "positive"
                        if previous_side == "left"
                        else "negative"
                    )
                    if motion_direction == self.in_direction:
                        self.interval_in += 1
                        self.net_count += 1
                    else:
                        self.interval_out += 1
                        self.net_count -= 1

                    with self.lock:
                        self.shared_dict[self.camera_id] = self.net_count
                        self.shared_totals[self.camera_id] = {
                            "in_count": self.interval_in,
                            "out_count": self.interval_out,
                        }
                        self.shared_count_ts[self.camera_id] = int(time.time() * 1000)
                    self.track_last_cross_ts[track_id] = now

                if curr_side != "middle":
                    state["last_non_middle_side"] = curr_side
                state["last_side"] = curr_side
                state["last_zone"] = curr_zone
                state["last_anchor_x"] = anchor_x
                state["last_anchor_y"] = anchor_y
            else:
                self.track_state[track_id] = {
                    "box": box,
                    "last_seen": now,
                    "age_frames": 1,
                    "last_side": curr_side,
                    "last_non_middle_side": curr_side if curr_side != "middle" else None,
                    "last_zone": curr_zone,
                    "last_anchor_x": anchor_x,
                    "last_anchor_y": anchor_y,
                }

        stale_ids = [
            track_id
            for track_id, state in self.track_state.items()
            if now - state["last_seen"] > self.TRACK_MEMORY_TIMEOUT_SEC
        ]
        for track_id in stale_ids:
            state = self.track_state.pop(track_id, None)
            if state is not None:
                self._remember_lost_track(track_id, state)
            self.track_last_cross_ts.pop(track_id, None)

        self._prune_lost_track_memory(now)

        boxes = []
        ids = []
        visible_track_ids = [
            track_id
            for track_id, state in self.track_state.items()
            if now - state["last_seen"] <= self.OVERLAY_TRACK_GRACE_SEC
            and not str(state.get("last_zone", "")).startswith("outside_")
        ]
        for track_id in sorted(visible_track_ids):
            state = self.track_state[track_id]
            x1, y1, x2, y2 = map(int, state["box"])
            boxes.append((x1, y1, x2, y2))
            ids.append(track_id)

        return boxes, ids

    def _draw_overlay(self, frame, boxes, ids):
        overlay = frame.copy()
        line_position = self._line_position()
        frame_h, frame_w = overlay.shape[:2]
        outer_left = line_position - self.OUTER_BAND
        outer_right = line_position + self.OUTER_BAND
        if self.count_axis == "y":
            cv2.line(
                overlay,
                (0, outer_left),
                (overlay.shape[1], outer_left),
                (255, 128, 0),
                1,
            )
            cv2.line(
                overlay,
                (0, outer_right),
                (overlay.shape[1], outer_right),
                (255, 128, 0),
                1,
            )
            cv2.line(
                overlay,
                (0, line_position),
                (overlay.shape[1], line_position),
                (0, 255, 255),
                2,
            )
            cv2.rectangle(
                overlay,
                (0, line_position - self.LINE_BAND),
                (overlay.shape[1], line_position + self.LINE_BAND),
                (255, 255, 0),
                1,
            )
        else:
            if self._use_perspective_counting():
                top_y = max(0, int(frame_h * self.perspective_far_y_ratio))
                bottom_y = max(0, frame_h - 1)
                top_outer_band = self._dynamic_outer_band(top_y, frame_h)
                bottom_outer_band = self._dynamic_outer_band(bottom_y, frame_h)
                top_line_band = self._dynamic_line_band(top_y, frame_h)
                bottom_line_band = self._dynamic_line_band(bottom_y, frame_h)

                cv2.line(
                    overlay,
                    (line_position - top_outer_band, top_y),
                    (line_position - bottom_outer_band, bottom_y),
                    (255, 128, 0),
                    1,
                )
                cv2.line(
                    overlay,
                    (line_position + top_outer_band, top_y),
                    (line_position + bottom_outer_band, bottom_y),
                    (255, 128, 0),
                    1,
                )
                cv2.line(
                    overlay,
                    (line_position, top_y),
                    (line_position, bottom_y),
                    (0, 255, 255),
                    2,
                )

                center_band = np.array(
                    [
                        (line_position - top_line_band, top_y),
                        (line_position + top_line_band, top_y),
                        (line_position + bottom_line_band, bottom_y),
                        (line_position - bottom_line_band, bottom_y),
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(overlay, [center_band], isClosed=True, color=(255, 255, 0), thickness=1)
            else:
                cv2.line(
                    overlay,
                    (outer_left, 0),
                    (outer_left, overlay.shape[0]),
                    (255, 128, 0),
                    1,
                )
                cv2.line(
                    overlay,
                    (outer_right, 0),
                    (outer_right, overlay.shape[0]),
                    (255, 128, 0),
                    1,
                )
                cv2.line(
                    overlay,
                    (line_position, 0),
                    (line_position, overlay.shape[0]),
                    (0, 255, 255),
                    2,
                )
                cv2.rectangle(
                    overlay,
                    (line_position - self.LINE_BAND, 0),
                    (line_position + self.LINE_BAND, overlay.shape[0]),
                    (255, 255, 0),
                    1,
                )

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            if max(0, x2 - x1) * max(0, y2 - y1) < self.MIN_BOX_AREA:
                continue
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
            if track_id is not None:
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
            f"Tiled YOLOv8 + tracker | {self._direction_hint()}",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return overlay

    def run(self):
        print(f"Camera {self.camera_id} started processing")
        process_thread = threading.Thread(target=self._processing_loop, daemon=True)
        process_thread.start()

        while True:
            if self.process_failed:
                return
            if not process_thread.is_alive():
                break
            time.sleep(0.05)

        process_thread.join(timeout=0.1)
