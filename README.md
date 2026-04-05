# Basketball Vision Analyzer 🏀

A high-performance, AI-driven tactical video analysis system for basketball. Featuring a **premium custom web UI**, **real-time tracking**, and **autonomous play-by-play commentary**.

---

## ⚡ Main Features

- **Custom YOLOv8 Tracking**: Specialized detection for players and the ball using fine-tuned `best.pt` weights.
- **Hybrid Ball Detection**: A three-tier hierarchy (YOLOv8 → HSV Fallback → Kalman Prediction) ensures the ball is never lost, even during high-speed shots or occlusions.
- **AI Commentary Engine**: Automated, rule-aware game analysis powered by **Gemini 1.5 Flash**.
    - **45-Frame Cycle**: Insightful commentary injected every 1.5 seconds.
    - **10-Word Constraint**: Concise, punchy descriptions for professional broadcast feel.
    - **Context Awareness**: Recognizes passes, speed bursts, and court zones (Paint, Wing, Backcourt).
- **Tactical HUD**: Dynamic team possession tracking, pass counting, and real-time player speed metrics.
- **Premium UI**: Single-page dark-mode interface with progress tracking and instant video downloads.

---

## 🛠️ Tech Stack

- **Computer Vision**: `OpenCV`, `YOLOv8` (Ultralytics), `Kalman Filters` (FilterPy)
- **AI & LLM**: `Google Gemini 1.5 Flash` (AI Commentary), `PyTorch`
- **Web Framework**: `FastAPI` (Backend), `Jinja2` (Templating)
- **Video Processing**: `MoviePy`, `FFmpeg`
- **Data Engineering**: `NumPy`, `SciPy`, `Tqdm`
- **Frontend**: Vanilla JS/CSS (Premium Dark-Mode UI)

---

## 🚀 Quick Start

### 1. Installation
Ensure you have Python 3.10+ and FFmpeg installed.
```bash
pip install -r requirements.txt
```

### 2. Launch the Analyzer
Run the single command below and open **[http://localhost:8000](http://localhost:8000)**:
```bash
python app.py
```
*(Windows users can also double-click `run_analyzer.ps1` to start instantly)*

---

## 📂 Project Structure

- `app.py`: FastAPI backend server managing uploads and processing jobs.
- `main.py`: Core processing pipeline logic.
- `core/`:
    - `detector.py`: YOLOv8 detection and jersey classification.
    - `tracker.py`: ByteTrack integration and Kalman/HSV ball tracking.
    - `physics.py`: Velocity estimation, court transformation, and event detection.
    - `homography.py`: 3D-to-2D court mapping.
- `utils/`:
    - `commentary_engine.py`: AI-driven commentary logic via Gemini API.
    - `visualizer.py`: HUD rendering, minimap, and tactical overlays.
- `configs/`: Court and gameplay configuration (`court_config.json`).
- `models/`: Weights files (e.g., `best.pt`).

---

## ⚙️ Configuration & CLI

### Court Customization
Modify `configs/court_config.json` to tune:
- **Jersey Classification**: Define team colors for the KMeans classifier.
- **Court Metrics**: Adjust baseline dimensions (NBA/FIBA) or manual corner points.
- **Physics**: Possession radius and speed thresholds.

### Advanced CLI Usage
For full-game analysis or developer testing, use `main.py` directly:
```bash
python main.py --video my_game.mp4 --out results.mp4 --max_frames 500
```
- `--video`: Path to input video (required).
- `--out`: Path to output analysis video (default: `result.mp4`).
- `--max_frames`: Limit analysis to a specific number of frames for fast testing.

---

## 📝 Analysis Limits
The web UI is optimized for clips up to **~5 minutes** to maintain peak performance on standard hardware. Large files (e.g., full 48-minute games) should be processed via the CLI for better resource management.
