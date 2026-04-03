from flask import Flask, request, jsonify, Response
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

# shared net counts from each camera
camera_net_counts = {}
camera_direction_totals = {}
camera_count_updated_at = {}  # Last time each camera count changed (ms epoch)
camera_running_status = {}  # Track which cameras are running
camera_frames = {}  # Latest raw + overlay JPEG frame bytes per camera
camera_density_stats = {}  # Latest crowd-density stats per camera
lock = Lock()

# Camera thread storage
camera_threads = {}
camera_workers = {}
camera_stream_paths = {}
camera_count_axes = {}
camera_in_directions = {}
camera_shot_types = {}
camera_processing_presets = {}
camera_counting_enabled = {}
camera_density_enabled = {}
camera_monitored_areas = {}


@app.route("/cameras", methods=["GET"])
def get_cameras():
    """Get list of all registered cameras"""
    cameras = []
    for cam_id in camera_threads.keys():
        cameras.append({
            "id": cam_id,
            "name": cam_id,
            "stream_path": camera_stream_paths.get(cam_id, "active"),
            "count_axis": camera_count_axes.get(cam_id, "x"),
            "in_direction": camera_in_directions.get(cam_id, "positive"),
            "shot_type": camera_shot_types.get(cam_id, "ground"),
            "processing_preset": camera_processing_presets.get(cam_id, "balanced"),
            "counting_enabled": camera_counting_enabled.get(cam_id, True),
            "density_enabled": camera_density_enabled.get(cam_id, False),
            "monitored_area_sqm": camera_monitored_areas.get(cam_id),
        })
    return jsonify(cameras)


@app.route("/cameras/add", methods=["POST"])
def add_camera():
    data = request.json
    name = data.get("name")
    stream_path = data.get("stream_path")
    count_axis = (data.get("count_axis") or "x").lower()
    in_direction = (data.get("in_direction") or "positive").lower()
    shot_type = (data.get("shot_type") or "ground").lower()
    processing_preset = (data.get("processing_preset") or "balanced").lower()
    counting_enabled = bool(data.get("counting_enabled", True))
    density_enabled = bool(data.get("density_enabled", False))
    monitored_area_sqm = data.get("monitored_area_sqm")

    if not name or not stream_path:
        return jsonify({"message": "Name and stream_path are required"}), 400

    if count_axis not in {"x", "y"}:
        return jsonify({"message": "count_axis must be 'x' or 'y'"}), 400
    if in_direction not in {"positive", "negative"}:
        return jsonify({"message": "in_direction must be 'positive' or 'negative'"}), 400
    if shot_type not in {"drone", "ground"}:
        return jsonify({"message": "shot_type must be 'drone' or 'ground'"}), 400
    if processing_preset not in {"fast", "balanced", "accurate"}:
        return jsonify({"message": "processing_preset must be 'fast', 'balanced', or 'accurate'"}), 400
    if density_enabled:
        try:
            monitored_area_sqm = float(monitored_area_sqm)
        except (TypeError, ValueError):
            return jsonify({"message": "monitored_area_sqm must be a positive number when density is enabled"}), 400
        if monitored_area_sqm <= 0:
            return jsonify({"message": "monitored_area_sqm must be a positive number when density is enabled"}), 400
    else:
        monitored_area_sqm = None

    if name in camera_threads:
        return jsonify({"message": "Camera already exists"}), 400

    # Start camera worker thread
    worker = CameraWorker(
        name,
        stream_path,
        count_axis,
        in_direction,
        shot_type,
        processing_preset,
        counting_enabled,
        density_enabled,
        monitored_area_sqm,
        camera_net_counts,
        camera_direction_totals,
        camera_count_updated_at,
        camera_frames,
        camera_density_stats,
        camera_running_status,
        lock,
    )

    thread = Thread(target=worker.run, daemon=True)
    thread.start()

    camera_workers[name] = worker
    camera_threads[name] = thread
    camera_stream_paths[name] = stream_path
    camera_count_axes[name] = count_axis
    camera_in_directions[name] = in_direction
    camera_shot_types[name] = shot_type
    camera_processing_presets[name] = processing_preset
    camera_counting_enabled[name] = counting_enabled
    camera_density_enabled[name] = density_enabled
    camera_monitored_areas[name] = monitored_area_sqm
    camera_running_status[name] = True  # Mark as running
    with lock:
        camera_net_counts[name] = 0
        camera_direction_totals[name] = {"in_count": 0, "out_count": 0}
        camera_count_updated_at[name] = int(time.time() * 1000)
        camera_density_stats[name] = {
            "density_enabled": density_enabled,
            "weighted_people": 0.0,
            "density_score": 0.0,
            "smoothed_density_score": 0.0,
            "monitored_area_sqm": monitored_area_sqm,
            "updated_at": int(time.time() * 1000),
        }

    return jsonify({
        "id": name,
        "name": name,
        "stream_path": stream_path,
        "count_axis": count_axis,
        "in_direction": in_direction,
        "shot_type": shot_type,
        "processing_preset": processing_preset,
        "counting_enabled": counting_enabled,
        "density_enabled": density_enabled,
        "monitored_area_sqm": monitored_area_sqm,
    }), 201


@app.route("/cameras/<camera_id>", methods=["DELETE"])
def delete_camera(camera_id):
    """Delete/stop a camera"""
    if camera_id not in camera_threads:
        return jsonify({"message": "Camera not found"}), 404

    # Remove from tracking
    del camera_threads[camera_id]
    camera_workers.pop(camera_id, None)
    camera_stream_paths.pop(camera_id, None)
    camera_count_axes.pop(camera_id, None)
    camera_in_directions.pop(camera_id, None)
    camera_shot_types.pop(camera_id, None)
    camera_processing_presets.pop(camera_id, None)
    camera_counting_enabled.pop(camera_id, None)
    camera_density_enabled.pop(camera_id, None)
    camera_monitored_areas.pop(camera_id, None)
    with lock:
        camera_net_counts.pop(camera_id, None)
        camera_direction_totals.pop(camera_id, None)
        camera_count_updated_at.pop(camera_id, None)
        camera_frames.pop(camera_id, None)
        camera_density_stats.pop(camera_id, None)

    return jsonify({"message": f"Camera {camera_id} deleted"})


@app.route("/cameras/counts", methods=["GET"])
def get_all_counts():
    """Get current counts for all cameras"""
    with lock:
        counts = []
        for cam_id, count in camera_net_counts.items():
            totals = camera_direction_totals.get(cam_id, {"in_count": 0, "out_count": 0})
            density = camera_density_stats.get(cam_id, {})
            counts.append({
                "camera_id": cam_id,
                "count": count,
                "in_count": totals.get("in_count", 0),
                "out_count": totals.get("out_count", 0),
                "timestamp": camera_count_updated_at.get(cam_id, int(time.time() * 1000)),
                "counting_enabled": camera_counting_enabled.get(cam_id, True),
                "density_enabled": density.get("density_enabled", camera_density_enabled.get(cam_id, False)),
                "weighted_people": density.get("weighted_people", 0.0),
                "density_score": density.get("density_score", 0.0),
                "smoothed_density_score": density.get("smoothed_density_score", 0.0),
                "monitored_area_sqm": density.get("monitored_area_sqm", camera_monitored_areas.get(cam_id)),
                "density_updated_at": density.get("updated_at"),
            })
    return jsonify(counts)


@app.route("/cameras/stream/<camera_id>", methods=["GET"])
def get_camera_stream(camera_id):
    """
    MJPEG stream endpoint.
    Query params:
      - overlay=true|false (default true)
    """
    overlay = request.args.get("overlay", "true").lower() != "false"
    key = "overlay" if overlay else "raw"

    if camera_id not in camera_threads:
        return jsonify({"message": "Camera not found"}), 404

    def frame_generator():
        last_ts = -1
        while True:
            with lock:
                frame_entry = camera_frames.get(camera_id)

            if not frame_entry:
                time.sleep(0.03)
                continue

            frame_ts = frame_entry.get("timestamp", 0)
            jpg_bytes = frame_entry.get(key)

            if not jpg_bytes:
                time.sleep(0.03)
                continue

            # Avoid re-sending exactly the same frame in a tight loop.
            if frame_ts == last_ts:
                time.sleep(0.01)
                continue

            last_ts = frame_ts
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )

    return Response(
        frame_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/cameras/counts/<camera_id>", methods=["GET"])
def get_camera_count(camera_id):
    """Get current count for a specific camera"""
    with lock:
        count = camera_net_counts.get(camera_id, 0)
        totals = camera_direction_totals.get(camera_id, {"in_count": 0, "out_count": 0})
        ts = camera_count_updated_at.get(camera_id, int(time.time() * 1000))
        density = camera_density_stats.get(camera_id, {})
    
    return jsonify({
        "camera_id": camera_id,
        "count": count,
        "in_count": totals.get("in_count", 0),
        "out_count": totals.get("out_count", 0),
        "timestamp": ts,
        "counting_enabled": camera_counting_enabled.get(camera_id, True),
        "density_enabled": density.get("density_enabled", camera_density_enabled.get(camera_id, False)),
        "weighted_people": density.get("weighted_people", 0.0),
        "density_score": density.get("density_score", 0.0),
        "smoothed_density_score": density.get("smoothed_density_score", 0.0),
        "monitored_area_sqm": density.get("monitored_area_sqm", camera_monitored_areas.get(camera_id)),
        "density_updated_at": density.get("updated_at"),
    })


@app.route("/cameras/reset/<camera_id>", methods=["POST"])
def reset_camera_count(camera_id):
    """Reset the count for a specific camera to 0"""
    if camera_id not in camera_threads:
        return jsonify({"message": "Camera not found"}), 404
    
    worker = camera_workers.get(camera_id)
    if worker:
        worker.reset_local_count()

    with lock:
        camera_net_counts[camera_id] = 0
        camera_direction_totals[camera_id] = {"in_count": 0, "out_count": 0}
        camera_count_updated_at[camera_id] = int(time.time() * 1000)
    
    return jsonify({
        "message": f"Count reset for camera {camera_id}",
        "camera_id": camera_id,
        "count": 0,
        "timestamp": camera_count_updated_at[camera_id]
    })


@app.route("/cameras/reset-all", methods=["POST"])
def reset_all_camera_counts():
    """Reset the count for all cameras to 0"""
    for worker in camera_workers.values():
        worker.reset_local_count()

    with lock:
        for cam_id in camera_net_counts:
            camera_net_counts[cam_id] = 0
            camera_direction_totals[cam_id] = {"in_count": 0, "out_count": 0}
            camera_count_updated_at[cam_id] = int(time.time() * 1000)
    
    return jsonify({
        "message": "All camera counts reset to 0",
        "timestamp": int(time.time() * 1000)
    })


@app.route("/cameras/start/<camera_id>", methods=["POST"])
def start_camera(camera_id):
    """Start monitoring for a specific camera"""
    if camera_id not in camera_threads:
        return jsonify({"message": "Camera not found"}), 404
    
    with lock:
        camera_running_status[camera_id] = True
    
    return jsonify({
        "message": f"Camera {camera_id} started",
        "camera_id": camera_id,
        "status": "running",
        "timestamp": int(time.time() * 1000)
    })


@app.route("/cameras/stop/<camera_id>", methods=["POST"])
def stop_camera(camera_id):
    """Stop monitoring for a specific camera"""
    if camera_id not in camera_threads:
        return jsonify({"message": "Camera not found"}), 404
    
    worker = camera_workers.get(camera_id)
    if worker:
        worker.reset_local_count()

    with lock:
        camera_running_status[camera_id] = False
        # Reset count when stopped
        camera_net_counts[camera_id] = 0
        camera_direction_totals[camera_id] = {"in_count": 0, "out_count": 0}
        camera_count_updated_at[camera_id] = int(time.time() * 1000)
    
    return jsonify({
        "message": f"Camera {camera_id} stopped",
        "camera_id": camera_id,
        "status": "stopped",
        "timestamp": camera_count_updated_at[camera_id]
    })


@app.route("/net_count", methods=["GET"])
def get_net_count():
    with lock:
        total_net = sum(camera_net_counts.values())

    return jsonify({
        "global_net_count": total_net,
        "total_people_inside": total_net,
        "per_camera": camera_net_counts,
        "per_camera_totals": camera_direction_totals,
        "timestamp": int(time.time() * 1000)
    })


@app.route("/update_count", methods=["POST"])
def update_count():
    data = request.json
    cam_id = data["camera_id"]
    net = data["net_count"]

    with lock:
        # Only update count if camera is running
        if camera_running_status.get(cam_id, True):
            camera_net_counts[cam_id] = net
            camera_count_updated_at[cam_id] = int(time.time() * 1000)

    return jsonify({"message": "Count updated"})


# =========================
# CREATE DB (ONLY ON START)
# =========================
with app.app_context():
    db.create_all()


if __name__ == "__main__":
    print("🚀 Central Server Running...")
    app.run(host="0.0.0.0", port=8000, debug=True)
