# 🚀 FYP Stampede Alert System — Setup Guide

Follow these steps to set up and run the project locally.

---

## 📥 Step 1: Clone the Repository

```bash
git clone "https://github.com/Nikhilesh-Satish/FYP-Stampede-Alert-System.git"
cd FYP-Stampede-Alert-System
```

---

## ⚙️ Step 2: Install Backend Dependencies

Run the following command from the project root:

```bash
pip install -r requirements.txt
```

---

## 🖥️ Step 3: Open a New Terminal

Keep the backend terminal open.

Navigate to the frontend folder:

```bash
cd Frontend
```

---

## 📦 Step 4: Install Frontend Dependencies

```bash
npm install
```

---

## ▶️ Step 5: Start the Frontend Server

```bash
npm run dev
```

---

## 🧠 Step 6: Start the Backend Server

Go back to the project root terminal and run:

```bash
python server.py
```

---

✅ The application should now be running successfully.

---

## 🎯 Fast Dense-Crowd YOLOv8 Workflow

1. Create tiled dataset from the original full-frame labels:

```bash
./venv/bin/python tile_yolo_dataset.py \
  --data crowd_data.yaml \
  --output dataset_tiled \
  --tile-size 512 \
  --overlap 0.40
```

2. Train on tiles with a faster two-stage schedule:

```bash
./venv/bin/python train_yolov8_crowd.py \
  --data dataset_tiled/data.yaml \
  --base yolov8m.pt \
  --stage1-epochs 15 \
  --stage2-epochs 20 \
  --imgsz 640 \
  --batch 8
```

3. Deploy the final checkpoint:

```bash
cp runs/detect/stampede_crowd_ft_stage2/weights/best.pt models/stampede_model.pt
```

4. Start backend with tiled live inference enabled:

```bash
export TILE_INFERENCE_ENABLED=true
export TILE_SIZE=640
export TILE_OVERLAP=0.35
python3 server.py
```
