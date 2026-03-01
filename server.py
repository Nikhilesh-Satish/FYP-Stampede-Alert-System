from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread, Lock
import time

from camera_worker import CameraWorker

# ✅ NEW: Auth imports
from auth.models import db
from auth.routes import auth_bp


app = Flask(__name__)

# ✅ Configure CORS properly
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ✅ NEW: Database config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# ✅ Register auth routes
app.register_blueprint(auth_bp, url_prefix='/auth')


# =========================
# EXISTING CAMERA SYSTEM
# =========================

# Shared net counts from each camera
camera_net_counts = {}
lock = Lock()

# Camera thread storage
camera_threads = {}


@app.route("/cameras", methods=["GET"])
def get_cameras():
    """Get list of all registered cameras"""
    cameras = []
    for cam_id in camera_threads.keys():
        cameras.append({
            "id": cam_id,
            "name": cam_id,
            "stream_path": "active"
        })
    return jsonify(cameras)


@app.route("/cameras/add", methods=["POST"])
def add_camera():
    data = request.json
    name = data.get("name")
    stream_path = data.get("stream_path")

    if not name or not stream_path:
        return jsonify({"message": "Name and stream_path are required"}), 400

    if name in camera_threads:
        return jsonify({"message": "Camera already exists"}), 400

    # Start camera worker thread
    worker = CameraWorker(name, stream_path, camera_net_counts, lock)

    thread = Thread(target=worker.run, daemon=True)
    thread.start()

    camera_threads[name] = thread

    return jsonify({
        "id": name,
        "name": name,
        "stream_path": stream_path
    }), 201


@app.route("/cameras/<camera_id>", methods=["DELETE"])
def delete_camera(camera_id):
    """Delete/stop a camera"""
    if camera_id not in camera_threads:
        return jsonify({"message": "Camera not found"}), 404

    # Remove from tracking
    del camera_threads[camera_id]
    with lock:
        camera_net_counts.pop(camera_id, None)

    return jsonify({"message": f"Camera {camera_id} deleted"})


@app.route("/cameras/counts", methods=["GET"])
def get_all_counts():
    """Get current counts for all cameras"""
    with lock:
        counts = []
        for cam_id, count in camera_net_counts.items():
            counts.append({
                "camera_id": cam_id,
                "count": count,
                "timestamp": int(time.time())
            })
    return jsonify(counts)


@app.route("/cameras/counts/<camera_id>", methods=["GET"])
def get_camera_count(camera_id):
    """Get current count for a specific camera"""
    with lock:
        count = camera_net_counts.get(camera_id, 0)
    
    return jsonify({
        "camera_id": camera_id,
        "count": count,
        "timestamp": int(time.time())
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


# =========================
# CREATE DB (ONLY ON START)
# =========================
with app.app_context():
    db.create_all()


if __name__ == "__main__":
    print("🚀 Central Server Running...")
    app.run(host="0.0.0.0", port=8000, debug=True)