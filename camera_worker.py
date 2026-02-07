import cv2
import time
from model_loader import load_model


class CameraWorker:
    def __init__(self, camera_id, video_path, shared_dict, lock):
        self.camera_id = camera_id
        self.video_path = video_path
        self.shared_dict = shared_dict
        self.lock = lock

        # Load trained YOLO model
        self.model = load_model()

        # Vertical counting line
        self.LINE_X = 320

        # Interval counters
        self.interval_in = 0
        self.interval_out = 0

        # Tracking memory
        self.last_x = {}
        self.counted_ids = set()

        # Timer reset
        self.last_reset_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)

        while True:
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            results = self.model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0]
            )

            # ✅ Safe detection check
            if results[0].boxes is None or results[0].boxes.id is None:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Optional ROI filter for accuracy
                if cy < 100 or cy > 400:
                    continue

                if track_id in self.last_x:
                    prev_x = self.last_x[track_id]

                    if track_id not in self.counted_ids:

                        if prev_x < self.LINE_X and cx >= self.LINE_X:
                            self.interval_in += 1
                            self.counted_ids.add(track_id)

                        elif prev_x > self.LINE_X and cx <= self.LINE_X:
                            self.interval_out += 1
                            self.counted_ids.add(track_id)

                self.last_x[track_id] = cx

            # Send every 10 seconds
            if time.time() - self.last_reset_time >= 10:
                net_count = self.interval_in - self.interval_out

                with self.lock:
                    self.shared_dict[self.camera_id] = net_count

                # Reset interval state
                self.interval_in = 0
                self.interval_out = 0
                self.counted_ids.clear()
                self.last_x.clear()

                self.last_reset_time = time.time()

            time.sleep(0.03)
