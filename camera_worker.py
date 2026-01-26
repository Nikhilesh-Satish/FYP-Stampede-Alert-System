
import cv2
import time
from ultralytics import YOLO

class CameraWorker:
    def __init__(self, camera_id, video_path, shared_dict, lock):
        self.camera_id = camera_id
        self.video_path = video_path
        self.shared_dict = shared_dict
        self.lock = lock

        self.model = YOLO("yolov8n.pt")

        self.LINE_X = 320
        self.in_count = 0
        self.out_count = 0

        self.last_x = {}
        self.counted_ids = set()

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

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2

                    if track_id in self.last_x:
                        prev_x = self.last_x[track_id]

                        if track_id not in self.counted_ids:
                            if prev_x < self.LINE_X and cx >= self.LINE_X:
                                self.in_count += 1
                                self.counted_ids.add(track_id)

                            elif prev_x > self.LINE_X and cx <= self.LINE_X:
                                self.out_count += 1
                                self.counted_ids.add(track_id)

                    self.last_x[track_id] = cx

            with self.lock:
                self.shared_dict[self.camera_id] = self.in_count - self.out_count

            time.sleep(0.03)  # simulate real-time FPS
