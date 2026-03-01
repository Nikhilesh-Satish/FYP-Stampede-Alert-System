import cv2
import time
from model_loader import load_model


class CameraWorker:
    def __init__(self, camera_id, video_path, shared_dict, lock):
        self.camera_id = camera_id
        self.video_path = video_path
        self.shared_dict = shared_dict
        self.lock = lock

        # ✅ Load your trained YOLO model
        self.model = load_model()

        # ✅ Horizontal counting line (middle of frame)
        self.LINE_Y = 240

        # ✅ Cumulative count of people inside (never goes below 0)
        self.people_inside = 0

        # ✅ Interval counters (reset every 10 seconds for reporting)
        self.interval_in = 0
        self.interval_out = 0

        # ✅ Tracking memory
        self.last_y = {}
        self.counted_ids = set()

        # ✅ Timer for 10-sec reset
        self.last_reset_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"❌ Camera {self.camera_id}: Failed to open video stream at '{self.video_path}'")
            return

        print(f"🎥 Camera {self.camera_id} started processing...")

        while True:
            ret, frame = cap.read()

            # Restart video if ended
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # ✅ Apply YOLO Tracking
            results = self.model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0]  # person class only
            )

            # ✅ If detections exist
            if results[0].boxes is not None and results[0].boxes.id is not None:

                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)

                    # ✅ Compute centroid Y
                    cy = (y1 + y2) // 2

                    # If we saw this ID before
                    if track_id in self.last_y:
                        prev_y = self.last_y[track_id]

                        # Count only once per ID per interval
                        if track_id not in self.counted_ids:

                            # ✅ DOWNWARD crossing → IN
                            if prev_y < self.LINE_Y and cy >= self.LINE_Y:
                                self.interval_in += 1
                                self.people_inside += 1
                                self.counted_ids.add(track_id)
                                print(f"{self.camera_id}: ID {track_id} -> IN (total: {self.people_inside})")

                            # ✅ UPWARD crossing → OUT
                            elif prev_y > self.LINE_Y and cy <= self.LINE_Y:
                                self.interval_out += 1
                                self.people_inside = max(0, self.people_inside - 1)  # Never go below 0
                                self.counted_ids.add(track_id)
                                print(f"{self.camera_id}: ID {track_id} -> OUT (total: {self.people_inside})")

                    # Update last position
                    self.last_y[track_id] = cy

            # ✅ Every 10 seconds → send net count to server
            if time.time() - self.last_reset_time >= 10:

                # Update shared dictionary safely with cumulative count
                with self.lock:
                    self.shared_dict[self.camera_id] = self.people_inside

                print(f"\n📡 {self.camera_id} REPORT (10 sec)")
                print(f"IN: {self.interval_in}, OUT: {self.interval_out}")
                print(f"PEOPLE INSIDE: {self.people_inside}\n")

                # ✅ Reset interval counters for next period
                self.interval_in = 0
                self.interval_out = 0
                self.counted_ids.clear()
                self.last_y.clear()

                self.last_reset_time = time.time()

            # Small delay for real-time simulation (~30 FPS)
            time.sleep(0.03)
