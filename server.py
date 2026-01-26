

from flask import Flask, request, jsonify
from multiprocessing import Process, Manager, Lock
import time
from camera_worker import CameraWorker

app = Flask(__name__)

manager = Manager()
camera_net_counts = manager.dict()
lock = Lock()
camera_processes = {}

@app.route("/add_camera", methods=["POST"])
def add_camera():
    data = request.json
    camera_id = data["camera_id"]
    video_path = data["video_path"]

    if camera_id in camera_processes:
        return jsonify({"message": "Camera already exists"}), 400

    worker = CameraWorker(camera_id, video_path, camera_net_counts, lock)
    process = Process(target=worker.run)
    process.start()

    camera_processes[camera_id] = process

    return jsonify({"message": f"Camera {camera_id} added and processing started"})


@app.route("/net_count", methods=["GET"])
def get_net_count():
    with lock:
        total_inside = sum(camera_net_counts.values())

    return jsonify({
        "total_people_inside": total_inside,
        "per_camera": dict(camera_net_counts),
        "timestamp": int(time.time())
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
