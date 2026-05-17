# Quick Start: HF Distributed System with Frames

## Prerequisites

✅ All done if you've deployed to HF Spaces

## Setup (30 seconds)

### 1. Install Dependencies

```bash
cd /Users/nik/Desktop/FYP-Stampede-Alert-System
pip install requests
```

### 2. Start Central Server

```bash
python server.py
```

You should see:

```
 * Running on http://0.0.0.0:8000
```

### 3. Test Round-Robin Distribution

**Add Camera 1** (will use HF Space 1):

```bash
curl -X POST http://localhost:8000/cameras/add \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gate1",
    "stream_path": "https://example.com/video1.mp4",
    "count_axis": "x",
    "in_direction": "positive",
    "shot_type": "ground"
  }'
```

**Monitor terminal** - you should see:

```
[gate1] Starting camera worker via HF API...
[gate1] Registered on HF API successfully
[gate1] Using API: https://Niks1904-stampede-alert-system-backend-fyp.hf.space
[gate1] Count: 0 (in: 0, out: 0)
[gate1] Frame updated (45230 bytes)
```

**Add Camera 2** (will use HF Space 2):

```bash
curl -X POST http://localhost:8000/cameras/add \
  -H "Content-Type: application/json" \
  -d '{
    "name": "entrance",
    "stream_path": "https://example.com/video2.mp4",
    "count_axis": "x",
    "in_direction": "positive",
    "shot_type": "ground"
  }'
```

You should see:

```
[entrance] Using API: https://Niks1904-stampede-alert-system-backend-fyp-2.hf.space
```

**Different API!** ✅ Round-robin working!

### 4. Check Aggregated Counts

```bash
curl http://localhost:8000/net_count
```

Response:

```json
{
  "global_net_count": 5,
  "per_camera": {
    "gate1": 3,
    "entrance": 2
  },
  "per_camera_totals": {
    "gate1": { "in_count": 2, "out_count": 1 },
    "entrance": { "in_count": 1, "out_count": 1 }
  },
  "timestamp": 1715881234567
}
```

### 5. View Live Video Stream with Bounding Boxes

In your browser, visit:

```
http://localhost:8000/cameras/stream/gate1
```

You'll see MJPEG stream with:

- ✅ Bounding boxes around people
- ✅ Live counting
- ✅ In/Out annotations

### 6. Run Frontend

In another terminal:

```bash
cd Frontend
npm run dev
```

Frontend will show:

- ✅ List of cameras
- ✅ Live video feeds with bounding boxes
- ✅ Real-time counts (in/out per camera)
- ✅ Global count aggregated

## Architecture Overview

```
Frontend (localhost:5173)
         ↓
Central Server (localhost:8000)
    ├─ Aggregates counts
    ├─ Serves MJPEG streams
    └─ Distributes cameras
         ↓
HF Spaces (12 × round-robin)
    ├─ HF Space 1: camera "gate1"
    ├─ HF Space 2: camera "entrance"
    ├─ HF Space 3: camera "hallway"
    └─ ... (cycles through)
```

## What's Happening

1. **Frontend** calls `/cameras/add` to add a camera
2. **Central Server** gets next HF Space (round-robin)
3. **Camera Worker** registers camera on that HF Space
4. **HF Space** processes video (draws boxes, counts people)
5. **Camera Worker** polls HF Space every 2 seconds for:
   - ✅ Counts (in/out)
   - ✅ Annotated frames (with bounding boxes)
6. **Central Server** aggregates data across all cameras
7. **Frontend** displays combined results

## Verify Each Part

### Is HF Space 1 running?

```bash
curl https://Niks1904-stampede-alert-system-backend-fyp.hf.space/cameras
# Should return: []
```

### Can camera register?

```bash
# Check central server logs for: "[gate1] Registered on HF API successfully"
```

### Are counts updating?

```bash
# Check central server logs for: "[gate1] Count: X (in: Y, out: Z)"
```

### Can frontend see frames?

```bash
# Browser: http://localhost:5173
# Should show camera card with live video
```

## Troubleshooting

### "Unable to add camera" in Frontend

1. Check central server is running: `python server.py`
2. Check HF Spaces are online (visit each URL in browser)
3. Check terminal for errors: `[camera_id] Error...`

### Counts showing "0" for all cameras

1. Is video.mp4 accessible to HF Spaces?
2. Check HF Space logs for inference errors
3. Try with a different video file

### No video frames in Frontend

1. Check `/cameras/stream/camera_id` endpoint works:
   ```bash
   curl http://localhost:8000/cameras/stream/gate1 > frame.jpg
   file frame.jpg  # Should be JPEG
   ```
2. If empty, HF Space isn't returning frames

### Can't add 2nd camera (stuck waiting)

1. First camera probably using all quota from HF Space 2
2. Check which HF Space got assigned to camera 2
3. Visit that HF Space to verify it's running
4. Increase poll timeout in `camera_worker_hf.py`

## Files Modified

- ✅ `camera_worker_hf.py` - Now fetches frames every 1 second
- ✅ `hf_api_client.py` - Round-robin distribution
- ✅ `server.py` - MJPEG stream endpoint
- ✅ `requirements.txt` - Added `requests`

## Next Steps

- [ ] Test with real videos
- [ ] Monitor frame quality/latency
- [ ] Add more HF Spaces if needed
- [ ] Deploy central server to cloud (Render)
- [ ] Deploy frontend to Netlify/Vercel

---

**Status:** Ready to test! Run `python server.py` and try adding cameras 🚀
