# HF Distributed System - Complete Setup

## ✅ What's Ready to Go

### New Files Created

1. **`hf_api_client.py`** (159 lines)
   - 12 HF Space URLs configured
   - Round-robin camera assignment
   - API endpoint management

2. **`camera_worker_hf.py`** (165 lines)
   - Registers camera on assigned HF Space
   - Polls counts every 2 seconds
   - Fetches annotated frames every 1 second
   - **Draws bounding boxes around people** ✅

3. **`QUICK_START_HF.md`**
   - Step-by-step setup guide
   - Test commands
   - Troubleshooting

### Modified Files

1. **`server.py`**
   - Now imports `hf_api_client` instead of local model
   - Uses `CameraWorkerHF` for distributed inference
   - Aggregates counts from all cameras
   - Serves MJPEG streams with bounding boxes

---

## 🚀 Quick Start

### Step 1: Install

```bash
pip install requests
```

### Step 2: Run Central Server

```bash
python server.py
```

Terminal output:

```
[camera_id] Starting camera worker via HF API...
[camera_id] Registered on HF API successfully
[camera_id] Using API: https://Niks1904-stampede-...hf.space
[camera_id] Count: 42 (in: 25, out: 17)
[camera_id] Frame updated (45230 bytes)
```

### Step 3: Add Cameras

```bash
# Camera 1 → HF Space 1
curl -X POST http://localhost:8000/cameras/add \
  -H "Content-Type: application/json" \
  -d '{"name":"gate1","stream_path":"video.mp4"}'

# Camera 2 → HF Space 2 (automatically)
curl -X POST http://localhost:8000/cameras/add \
  -H "Content-Type: application/json" \
  -d '{"name":"entrance","stream_path":"video2.mp4"}'
```

### Step 4: View Results

```bash
# See all cameras
curl http://localhost:8000/cameras

# View live stream with bounding boxes
# Browser: http://localhost:8000/cameras/stream/gate1

# Get aggregated counts
curl http://localhost:8000/net_count
```

### Step 5: Run Frontend

```bash
cd Frontend
npm run dev
```

Frontend shows:

- ✅ List of cameras
- ✅ Live video with bounding boxes
- ✅ In/Out counts per camera
- ✅ Global count (sum of all cameras)

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (React/Vite)                   │
│              Displays cameras, counts, video                │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/HTTPS
                         ↓
┌─────────────────────────────────────────────────────────────┐
│            CENTRAL SERVER (Flask on localhost:8000)         │
├─────────────────────────────────────────────────────────────┤
│ • /cameras/add           → Register camera (assigns HF API) │
│ • /cameras               → List all cameras                 │
│ • /cameras/counts        → Get all counts                   │
│ • /cameras/stream/<id>   → MJPEG with bounding boxes       │
│ • /net_count             → Global aggregated count          │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
         ╔═══════════════╩═══════════════╗
         ↓                               ↓
    ┌─────────┐                    ┌─────────┐
    │ HF API  │                    │ HF API  │
    │ Space 1 │                    │ Space 2 │
    ├─────────┤                    ├─────────┤
    │ Camera1 │                    │ Camera2 │
    │ • Detect│                    │ • Detect│
    │ • Track │                    │ • Track │
    │ • Draw  │                    │ • Draw  │
    │ • Count │                    │ • Count │
    └─────────┘                    └─────────┘
         ↑                               ↑
         └───────────────┬───────────────┘
                         │ (12 × HF Spaces)
                   Poll every 2 sec for:
                   • Counts (in/out)
                   • Frames (boxes drawn)
```

---

## 🔄 Round-Robin Distribution

Each time you add a camera:

```
Camera 1 (gate1)    → HF Space 1  (Niks1904-...fyp)
Camera 2 (entrance) → HF Space 2  (Niks1904-...fyp-2)
Camera 3 (hallway)  → HF Space 3  (Niks1904-...fyp-3)
...
Camera 12 (storage) → HF Space 12 (Niks001904-...fyp-12)
Camera 13 (new)     → HF Space 1  (cycles back)
```

**Why?** Free tier HF Spaces have rate limits. By using 12 spaces, you get 12× capacity!

---

## 🎯 What Each Component Does

### HF API Client (`hf_api_client.py`)

```python
hf_client = HFAPIClient()
api_url = hf_client.assign_camera_api("gate1")  # Gets next space
hf_client.add_camera("gate1", "video.mp4")      # Register on space
counts = hf_client.get_camera_counts("gate1")   # Fetch counts
```

### Camera Worker (`camera_worker_hf.py`)

```python
worker = CameraWorkerHF(...)
worker.run()  # Main loop:
              # 1. Register on HF Space
              # 2. Poll counts every 2 sec
              # 3. Fetch frames every 1 sec
              # 4. Update shared state
```

### Server (`server.py`)

```python
@app.route("/cameras/add", methods=["POST"])  # Add camera
@app.route("/net_count", methods=["GET"])     # Get global count
@app.route("/cameras/stream/<id>", ...)       # Stream with boxes
```

---

## 📈 Data Flow

### Adding a Camera

```
Frontend
    ↓
POST /cameras/add {name: "gate1", stream_path: "video.mp4"}
    ↓
Central Server
    ↓
HFClient assigns → HF Space 1
    ↓
CameraWorkerHF.run()
    ↓
POST to HF Space 1: /cameras/add
    ↓
HF Space 1 processes video
    ↓
Worker polls HF Space 1 every 2 sec:
    ├─ GET /cameras/counts/gate1 → {count: 42, in: 25, out: 17}
    └─ GET /cameras/stream/gate1 → JPEG with bounding boxes
    ↓
Central Server aggregates
    ↓
Frontend displays
```

---

## ✨ Features Enabled

| Feature              | Status | How It Works                                   |
| -------------------- | ------ | ---------------------------------------------- |
| **Multiple Cameras** | ✅     | Round-robin assigns to different HF Spaces     |
| **Live Counting**    | ✅     | Polls every 2 seconds, updates in real-time    |
| **Bounding Boxes**   | ✅     | HF Space draws boxes, fetched every 1 sec      |
| **Aggregation**      | ✅     | Central server sums counts from all cameras    |
| **MJPEG Stream**     | ✅     | Flask serves stream endpoint with boxes        |
| **Frontend Display** | ✅     | Shows video + counts per camera + global total |

---

## 🔧 Customization

### Add More HF Spaces

Edit `hf_api_client.py`:

```python
HF_SPACES = [
    "https://huggingface.co/spaces/...", # Add new space here
    ...
]
```

### Change Poll Intervals

Edit `camera_worker_hf.py`:

```python
# Poll counts every N seconds
if now - last_poll > 2:  # Change 2 to desired value

# Fetch frame every N seconds
if now - last_frame_fetch > 1:  # Change 1 to desired value
```

### Change Aggregation Logic

Edit `server.py` `/net_count` route:

```python
# Add weights, filters, thresholds, etc.
total_net = sum(camera_net_counts.values())
```

---

## 🐛 Common Issues & Fixes

### "Unable to add camera"

```bash
# Check server is running
curl http://localhost:8000/cameras

# Check HF Space 1 is accessible
curl https://Niks1904-stampede-alert-system-backend-fyp.hf.space/cameras

# Check logs for errors
python server.py  # Look for [camera_id] errors
```

### Counts stuck at 0

```bash
# HF Space might not have video processing video
# Verify video file is accessible from HF
# Check HF Space logs for inference errors
```

### No frames showing

```bash
# Test MJPEG endpoint manually
curl http://localhost:8000/cameras/stream/gate1 > test.jpg

# If empty, frame fetch is failing
# Check logs for: "Error fetching frame"
```

### 2nd camera can't be added

```bash
# HF Space 2 might be busy
# Try waiting 10 seconds and retrying
# Or use different video stream for camera 2
```

---

## 📋 Deployment Checklist

- [ ] Test locally first with `python server.py`
- [ ] Verify all 12 HF Spaces are running
- [ ] Test adding 3+ cameras (verify round-robin)
- [ ] Check video frames display in `/cameras/stream/camera_id`
- [ ] Verify counts update every 2 seconds
- [ ] Test frontend display of cameras
- [ ] Deploy central server to cloud (Render, Heroku, etc.)
- [ ] Update Frontend API URL to cloud server
- [ ] Deploy frontend to Netlify/Vercel

---

## 🚀 Next: Deploy to Cloud

### Deploy Central Server

```bash
# Option 1: Render (free tier)
# Push repo, set Root Directory to root, Start Command:
# gunicorn server:app

# Option 2: Railway
# Similar setup, $5/month free credits

# Option 3: Heroku (paid)
```

### Update Frontend

```bash
# Frontend/.env
VITE_API_URL=https://your-central-server.onrender.com
```

### Deploy Frontend

```bash
# Netlify / Vercel
# Connect repo, auto-deploy
```

---

**Ready to test? Run:** `python server.py` 🎬
