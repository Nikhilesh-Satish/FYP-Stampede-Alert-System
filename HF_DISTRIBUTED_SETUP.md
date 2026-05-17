# Stampede Alert System - HF API Distributed Architecture

This is the updated setup that uses **multiple Hugging Face Space instances** for distributed inference via round-robin load balancing.

## Architecture

```
Central Server (localhost or Render)
    ↓
HF API Client (round-robin distribution)
    ↓ Assigns cameras to different spaces
    ├─ Camera 1 → HF Space 1
    ├─ Camera 2 → HF Space 2
    ├─ Camera 3 → HF Space 3
    └─ ... (cycles through 12 spaces)
    ↓
Aggregates counts from all cameras
    ↓
Frontend displays global count
```

## How It Works

### 1. **HF API Client** (`hf_api_client.py`)

- Manages 12 HF Space URLs
- Converts Space URLs to API endpoints
- Distributes cameras in **round-robin** fashion
- Tracks which API each camera uses

### 2. **Camera Worker** (`camera_worker_hf.py`)

- Registers camera on assigned HF API
- Polls HF API every 2 seconds for counts
- Updates shared dictionaries (for central aggregation)
- **No local model loading** - all inference on HF

### 3. **Server** (`server.py`)

- Central aggregation point
- Collects counts from all cameras
- Provides `/net_count` endpoint for global count
- Frontend shows combined totals

## HF Spaces

```
https://huggingface.co/spaces/Niks1904/stampede-alert-system-backend-fyp
https://huggingface.co/spaces/Niks1904/stampede-alert-system-backend-fyp-2
... (12 total spaces)
```

Each space runs a Flask backend with:

- `/cameras/add` - Register camera
- `/cameras/counts/<camera_id>` - Get counts
- `/cameras/<camera_id>` - Delete camera

## Setup

### 1. Install Dependencies

```bash
pip install requests flask numpy opencv-python
```

### 2. Run Central Server

```bash
# Local development
python server.py

# Production (Render)
gunicorn server:app
```

### 3. Test HF Integration

```bash
# Add a camera (will be assigned to HF Space 1)
curl -X POST http://localhost:8000/cameras/add \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gate1",
    "stream_path": "video.mp4",
    "count_axis": "x"
  }'

# Add another camera (will be assigned to HF Space 2)
curl -X POST http://localhost:8000/cameras/add \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gate2",
    "stream_path": "video2.mp4",
    "count_axis": "x"
  }'

# Get global count (aggregated from all cameras)
curl http://localhost:8000/net_count
```

## Round-Robin Distribution

**Example sequence:**

```
Camera 1 (gate1)     → HF Space 1
Camera 2 (entrance)  → HF Space 2
Camera 3 (hallway)   → HF Space 3
Camera 4 (exit)      → HF Space 4
Camera 5 (parking)   → HF Space 5
Camera 6 (lobby)     → HF Space 6
Camera 7 (roof)      → HF Space 7
... (cycles back to Space 1 for camera 13)
```

## API Response Format

Each HF Space returns counts like:

```json
{
  "camera_id": "gate1",
  "count": 42,
  "in_count": 25,
  "out_count": 17,
  "timestamp": 1715881234567
}
```

Central server aggregates to:

```json
{
  "global_net_count": 120,
  "per_camera": {
    "gate1": 42,
    "entrance": 35,
    "hallway": 43
  },
  "per_camera_totals": {
    "gate1": {"in_count": 25, "out_count": 17},
    "entrance": {"in_count": 20, "out_count": 15},
    ...
  }
}
```

## Troubleshooting

### Camera gets stuck at "waiting for data"

**Check:**

1. Is the HF Space running? Visit it in browser to verify
2. Is `camera_worker.py` polling correctly?
3. Is the video stream accessible from HF Space?

**Fix:**

```bash
# Check HF Space logs
# Visit: https://<user>-<space-name>.hf.space/logs

# Restart service
python server.py --restart
```

### Can't add 2nd camera

**Issue:** First HF Space may be overloaded

**Solution:**

1. Check which Space is assigned to Camera 1
2. Verify next Space (Camera 2) is responsive
3. Try adding Camera 2 to a different Space manually
4. Increase polling interval in `camera_worker_hf.py` (currently 2 seconds)

### Bounding boxes not showing

**Current limitation:** HF Spaces don't return frame images, only counts.

**Solution:** In frontend, display:

- Counts (in/out)
- Global density
- Camera-by-camera breakdown

Bounding boxes require frame data from inference APIs.

## Files Overview

| File                  | Purpose                                      |
| --------------------- | -------------------------------------------- |
| `hf_api_client.py`    | HF Space URL management, round-robin logic   |
| `camera_worker_hf.py` | Camera worker using HF APIs (no local model) |
| `server.py`           | Central aggregation server (Flask)           |
| `requirements.txt`    | Python dependencies                          |

## Performance

- **Latency:** ~2-4 sec (2 sec poll interval + inference time)
- **Cost:** Free HF tier (12 × free spaces)
- **Scalability:** Add more HF Spaces and update `HF_SPACES` list in `hf_api_client.py`

## Next Steps

1. ✅ Deploy each HF Space (already done)
2. ✅ Test round-robin distribution
3. ✅ Verify counts aggregation
4. Add frontend visualization for counts/totals
5. (Optional) Add frame streaming from HF Spaces for bounding box display

---

**Status:** Ready for testing with multiple cameras! 🚀
