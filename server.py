from flask import Flask, request, jsonify
from threading import Thread, Lock
import time

from camera_worker import CameraWorker

app = Flask(__name__)

# Shared net counts from each camera
camera_net_counts = {}
lock = Lock()

# Camera thread storage
camera_threads = {}


@app.route("/add_camera", methods=["POST"])
def add_camera():
    data = request.json
    camera_id = data["camera_id"]
    video_path = data["video_path"]

    if camera_id in camera_threads:
        return jsonify({"message": "Camera already exists"}), 400

    # Start camera worker thread
    worker = CameraWorker(camera_id, video_path, camera_net_counts, lock)

    thread = Thread(target=worker.run, daemon=True)
    thread.start()

    camera_threads[camera_id] = thread

    return jsonify({
        "message": f"Camera {camera_id} started successfully"
    })


@app.route("/net_count", methods=["GET"])
def get_net_count():
    with lock:
        total_inside = sum(camera_net_counts.values())

    return jsonify({
        "total_people_inside": total_inside,
        "per_camera": camera_net_counts,
        "timestamp": int(time.time())
    })

@app.route("/update_count", methods=["POST"])
def update_count():
    data = request.json
    cam_id = data["camera_id"]
    net = data["net_count"]

    with lock:
        camera_net_counts[cam_id] = net

    return jsonify({"message": "Count updated"})



if __name__ == "__main__":
    print("🚀 Central Server Running...")
    app.run(host="0.0.0.0", port=5000, debug=True)
