#  YOLO Face Detection — Streamlit App

Real-time multi-face detection powered by **YOLOv8** and **Streamlit WebRTC**.

## Features

-  **Live Webcam Detection** — Real-time face detection via browser webcam
-  **FPS Tracking** — Smoothed rolling-average FPS counter
-  **Multi-Face Handling** — Detects and scores all visible faces simultaneously
-  **Privacy Mode** — Toggle face blurring while still counting detections
-  **Session Logging** — Timestamped detection log with confidence and FPS
-  **CSV Export** — Download session data for analysis
-  **Premium Dark UI** — Glassmorphism dashboard with live metrics.

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Deploy!

> **Note:** Streamlit Cloud runs on CPU only. For best performance, place an
> ONNX-optimized model in the `models/` directory.

## Using a Custom Model

Place your trained model file in the `models/` directory:

```
models/
├── best.pt          # Ultralytics .pt format
└── yolov8n_face.onnx  # ONNX format (preferred for CPU)
```

If no model is found locally, the app will auto-download `yolov8n.pt` from Ultralytics.

## Project Structure

```
├── app.py              # Streamlit entry point
├── core/
│   ├── detector.py     # YOLO inference (ONNX + .pt dual-backend)
│   └── tracker.py      # FPS calculator + session data logger
├── ui/
│   ├── styles.py       # Custom dark-theme CSS
│   └── components.py   # Sidebar, metrics, and log components
├── models/             # Model weights (not tracked in git)
├── inference.py        # Original Colab notebook script
├── requirements.txt    # Python dependencies
├── packages.txt        # System deps for Streamlit Cloud
└── .streamlit/
    └── config.toml     # Theme configuration
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| AI Model | YOLOv8n (Ultralytics) |
| Inference | ONNX Runtime / PyTorch |
| Frontend | Streamlit + WebRTC |
| Vision | OpenCV |
| Styling | Custom CSS (Glassmorphism) |
