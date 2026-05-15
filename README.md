# 👷 Worker Uniform Colour Detection

A real-time computer vision tool that detects workers in a video and checks whether they are wearing a **blue uniform** using YOLOv8 and OpenCV.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run on a Video File

```bash
python worker_colour.py --video path/to/video.mp4
```

### 3. Run on Webcam

```bash
python worker_colour.py --video 0
```

### 4. Use a Custom YOLO Model

```bash
python worker_colour.py --video video.mp4 --model yolov8s.pt
```

---

## 🧠 How It Works

```
Video Frame
    ↓
[Step 1] Person Detection (YOLOv8)
    • Detects all people in the frame
    • Filters detections below 0.4 confidence

    ↓
[Step 2] HSV Colour Analysis (OpenCV)
    • Crops each detected person's region
    • Converts to HSV colour space
    • Applies blue colour mask (Hue: 90–130)

    ↓
[Step 3] Classification
    • Blue pixels > 5000 → Blue Uniform ✅ (blue box)
    • Blue pixels ≤ 5000 → No Uniform ❌ (red box)

    ↓
[Step 4] Display
    • Bounding boxes drawn on frame
    • Live count shown on screen
```

---

## 🎨 Blue Uniform HSV Range

| Parameter  | Lower | Upper |
|------------|-------|-------|
| Hue        | 90    | 130   |
| Saturation | 50    | 255   |
| Value      | 50    | 255   |

---

## 🗂️ File Structure

```
worker-colour-github/
├── worker_colour.py   ← Main detection script
├── requirements.txt   ← Python dependencies
└── README.md          ← This file
```

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `q` | Quit   |

---

## 📦 Tech Stack

| Component       | Technology          |
|-----------------|---------------------|
| Person Detection| YOLOv8 (Ultralytics)|
| Colour Analysis | OpenCV (HSV)        |
| Array Processing| NumPy               |
