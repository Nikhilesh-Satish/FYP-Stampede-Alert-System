# Deploy Backend to Hugging Face Spaces

Use this when Render free runs out of memory with `yolov8m.pt`.

## Create the Space

1. Go to Hugging Face Spaces and create a new Space.
2. Choose **Docker** as the SDK.
3. Use public visibility for the easiest frontend access.
4. Push the contents of the `backend/` folder to the Space repo root.

Example local push:

```bash
git clone https://huggingface.co/spaces/<username>/<space-name> hf-space
rsync -av --delete backend/ hf-space/
cd hf-space
git add .
git commit -m "Deploy stampede alert backend"
git push
```

The Space repo root should contain:

```text
Dockerfile
README.md
requirements.txt
server.py
wsgi.py
camera_worker.py
model_loader.py
auth/
bytetrack_crowd.yaml
yolov8m.pt
```

## Environment Variables

Set these in the Space settings if you want to override the Docker defaults:

```env
STAMPEDE_MODEL_PATH=yolov8m.pt
DEPLOYMENT_PROFILE=free
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

## Frontend API URL

After the Space builds, update the frontend environment variable:

```env
VITE_API_URL=https://<username>-<space-name>.hf.space
```

Then redeploy the frontend.

## Smoke Test

Open:

```text
https://<username>-<space-name>.hf.space/health
```

Expected response:

```json
{"status":"healthy"}
```

Then test:

```text
https://<username>-<space-name>.hf.space/cameras
```

Expected first response:

```json
[]
```

## Notes

- Free CPU Spaces can sleep when inactive, so the first request after idle time can be slow.
- `yolov8m.pt` is kept as the required model.
- For multiple active cameras, start with one Space backend and measure. If CPU becomes the bottleneck, duplicate worker Spaces and assign one camera per worker.
