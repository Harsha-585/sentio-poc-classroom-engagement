# Classroom Engagement Heatmap & Group Analysis

**Author:** Harshavardhan Perla
**Roll Number:** 25CS60R72
**Institution:** IIT Kharagpur

---

## 📌 Overview

This project analyzes classroom engagement from video recordings using computer vision and AI. It generates:
- **Spatial engagement heatmap** (4×6 grid)
- **Time-series engagement trends** (6-second windows)
- **Gaze direction analysis** (forward/down/left/right)
- **Interactive HTML dashboard** with visualizations
- **Integration-ready JSON output**

---

## 🎯 Key Features

✅ **Face Detection:** MediaPipe-based multi-face detection (up to 50 faces)
✅ **Spatial Zoning:** Divides classroom into 4 rows × 6 columns (24 zones)
✅ **Engagement Scoring:** Real-time scoring based on gaze direction and face visibility
✅ **Window Analysis:** 6-second temporal windows with aggregated metrics
✅ **Evidence Collection:** Identifies and captures 3 worst-performing windows
✅ **Offline Dashboard:** Self-contained HTML with embedded charting (no CDN dependencies)
✅ **JSON Export:** Standardized integration format for external systems

---

## 🏗️ Architecture

### System Pipeline

```
Video Input
    ↓
Frame Extraction (OpenCV)
    ↓
Face Detection (MediaPipe)
    ↓
Spatial Zone Mapping (4×6 Grid)
    ↓
Gaze Estimation (Facial Landmarks)
    ↓
Engagement Scoring (Per Face → Per Zone)
    ↓
Window Aggregation (6-second chunks)
    ↓
Dashboard Generation (HTML + Charts)
    ↓
JSON Export (Integration Format)
```

### Core Components

1. **ClassroomEngagementAnalyzer**
   - `analyze_video()`: Main analysis pipeline
   - `get_zone_id()`: Maps (x, y) coordinates to grid zones
   - `estimate_gaze_direction()`: Classifies gaze from landmarks
   - `calculate_engagement_score()`: Computes 0-100 engagement score

2. **Visualization Module**
   - `create_dashboard()`: Generates HTML report
   - Line chart: Engagement over time
   - Heatmap: Zone-wise engagement visualization
   - Evidence cards: Worst 3 windows with thumbnails
   - Bar chart: Gaze direction distribution

3. **Integration Module**
   - `save_integration_json()`: Exports structured JSON

---

## 📋 Requirements

### System Requirements
- **Python:** 3.9 or higher
- **OS:** Linux, macOS, or Windows
- **RAM:** 4 GB minimum (8 GB recommended for large videos)
- **Storage:** 500 MB for dependencies + video size

### Python Dependencies

```txt
opencv-python==4.9.0
mediapipe==0.10.14
numpy==1.26.4
Pillow==10.3.0
```

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd sentio-poc-classroom-engagement
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you encounter installation issues with MediaPipe on ARM-based Macs (M1/M2), use:
```bash
pip install mediapipe-silicon
```

---

## 💻 Usage

### Basic Execution

```bash
python solution.py <path-to-video>
```

**Example:**
```bash
python solution.py classroom_recording.mp4
```

### Expected Output

```
============================================================
CLASSROOM ENGAGEMENT ANALYSIS
============================================================
Processing video: classroom_recording.mp4

Video stats: 30 FPS, 9000 frames, 1920x1080
Processing 180 frames per 6s window...
Processed 3000/9000 frames...
Processed 6000/9000 frames...
Processed 9000/9000 frames...

Analysis complete! Processed 50 windows.
Average engagement: 67.3

Step 2/3: Creating dashboard...
Dashboard created: engagement_report.html

Step 3/3: Saving integration JSON...
Integration JSON saved: engagement_output.json

============================================================
✅ ANALYSIS COMPLETE!
============================================================

Generated files:
  📊 engagement_report.html - Interactive dashboard
  📄 engagement_output.json - Integration data

Open engagement_report.html in your browser to view the results.
```

### Generated Files

1. **`engagement_report.html`**
   - Self-contained HTML dashboard
   - No internet connection required
   - Includes 4 visualizations:
     - Engagement line chart
     - Spatial heatmap (4×6 grid)
     - Worst 3 windows with thumbnails
     - Gaze distribution bar chart

2. **`engagement_output.json`**
   - Integration-ready JSON with schema:
   ```json
   {
     "session_id": "classroom_001",
     "total_windows": 50,
     "windows": [
       {
         "window_id": 1,
         "timestamp": 0,
         "group_score": 68.5,
         "zone_engagement": [
           {"zone_id": "R1C1", "engagement_score": 72.3},
           ...
         ],
         "dominant_gaze_distribution": {
           "forward": 45.2,
           "down": 30.1,
           "left": 12.5,
           "right": 12.2
         }
       },
       ...
     ]
   }
   ```

---

## 🧪 Testing

### Test with Sample Video

If you have a test video, run:

```bash
python solution.py test_video.mp4
```

Verify outputs:
1. Check that `engagement_report.html` opens correctly in a browser
2. Validate that JSON schema matches integration requirements
3. Ensure heatmap displays 4×6 grid properly
4. Confirm worst 3 windows show thumbnails

### Expected Behavior

- **For videos with no faces:** Engagement scores will be 0 across all zones
- **For stationary classroom:** Gaze will predominantly show "forward"
- **For dynamic scenes:** Expect variation in zone scores and gaze directions

---

## 📊 Methodology

### Engagement Scoring Algorithm

```python
# Per-Face Engagement
gaze_weight = {
    "forward": 1.0,   # Fully engaged
    "down": 0.4,      # Writing/looking down
    "left": 0.6,      # Distracted
    "right": 0.6      # Distracted
}

face_engagement = (gaze_score * 0.7 + visibility * 0.3) * 100
```

### Zone Aggregation

```python
# Zone Score = Average of all faces in that zone
zone_score = mean([face_1_score, face_2_score, ...])

# Group Score = Average of all non-zero zones
group_score = mean([zone_scores where score > 0])
```

### Gaze Classification

Uses facial landmark ratios:
- **Nose tip** vs. **eye center** horizontal offset → Left/Right
- **Nose tip** vs. **face center** vertical offset → Up/Down
- Thresholds: `h_threshold=0.03`, `v_threshold=0.04`

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'mediapipe'"

**Solution:** Install MediaPipe:
```bash
pip install mediapipe==0.10.14
```

### Issue: Video not processing / "File not found"

**Solution:** Check file path and format. Supported formats:
- `.mp4` (recommended)
- `.avi`
- `.mov`
- `.mkv`

### Issue: "Killed" or out-of-memory errors

**Solution:** Reduce video resolution or duration:
```bash
# Use FFmpeg to resize
ffmpeg -i input.mp4 -vf scale=1280:720 output.mp4
```

### Issue: HTML dashboard not showing charts

**Solution:** Ensure JavaScript is enabled in your browser. Try opening in:
- Chrome (recommended)
- Firefox
- Safari

---

## 📁 Project Structure

```
senito-poc-classroom-engagement/
│
├── solution.py                  # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── engagement_report.html       # Generated dashboard (after running)
├── engagement_output.json       # Generated JSON (after running)
│
└── sample_videos/              # Place test videos here (optional)
    └── classroom_sample.mp4
```

---

## 🎥 Demo Video Guidelines

To record a demo video (max 2 minutes):

1. **Introduction (15s)**
   - Show project structure
   - Highlight key files

2. **Execution (45s)**
   - Run `python solution.py <video>`
   - Show terminal output

3. **Results (45s)**
   - Open `engagement_report.html`
   - Navigate through visualizations
   - Show JSON structure

4. **Conclusion (15s)**
   - Summarize key features

**Recording tips:**
- Use screen recording software (OBS Studio, QuickTime, Windows Game Bar)
- Export as `.mp4` at 1080p
- Keep file size under 50 MB

---

## 🧠 Technical Decisions

### Why MediaPipe over face_recognition?

- **Performance:** MediaPipe is GPU-accelerated and 3-5x faster
- **Landmarks:** Provides 468 landmarks vs. 68 (better gaze estimation)
- **Scalability:** Handles up to 50 faces simultaneously

### Why Embedded Chart.js?

- **Offline support:** No CDN dependencies
- **Lightweight:** ~3KB for minimal implementation
- **Compatibility:** Works across all modern browsers

### Why 6-second windows?

- **Balance:** Captures behavioral patterns while maintaining temporal granularity
- **Standard:** Aligns with educational research on attention spans
- **Performance:** Reasonable processing time for real-time applications

---

## 📝 Assignment Compliance Checklist

✅ **solution.py** with fixed function signatures
✅ **engagement_report.html** with 4 required visualizations
✅ **engagement_output.json** matching fixed schema
✅ **README.md** with comprehensive documentation
✅ **requirements.txt** with all dependencies
✅ Python 3.9+ (no notebooks)
✅ Offline HTML (no CDN)
✅ Private repository with Sentiodirector as collaborator
✅ Branch: `Harshavardhan_Perla_25CS60R72`

---

## 🤝 Acknowledgments

- **Sentio Mind** for the problem statement
- **IIT Kharagpur CDC** for coordination
- **MediaPipe** (Google) for face detection library
- **Chart.js** for visualization inspiration

---

## 📬 Contact

**Harshavardhan Perla**
Roll Number: 25CS60R72
IIT Kharagpur

---

## 📄 License

This project is submitted as part of an academic assessment. All rights reserved.

---

**Last Updated:** March 25, 2026
**Version:** 1.0
