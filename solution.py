import cv2
import numpy as np
import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import BytesIO
from PIL import Image

# Try different face detection libraries
FACE_DETECTION_METHOD = None
face_mesh = None

# Method 1: Try MediaPipe
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=50,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    FACE_DETECTION_METHOD = "mediapipe"
    print("✅ Using MediaPipe for face detection")
except Exception as e:
    print(f"⚠️  MediaPipe failed: {e}")

# Method 2: Try face_recognition as backup
if FACE_DETECTION_METHOD is None:
    try:
        import face_recognition
        FACE_DETECTION_METHOD = "face_recognition"
        print("✅ Using face_recognition library as backup")
    except Exception as e:
        print(f"⚠️  face_recognition failed: {e}")

# Method 3: OpenCV face detection as final fallback
if FACE_DETECTION_METHOD is None:
    try:
        # Download OpenCV face detection model if needed
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        FACE_DETECTION_METHOD = "opencv"
        print("✅ Using OpenCV Haar Cascades as fallback")
    except Exception as e:
        print(f"❌ All face detection methods failed: {e}")
        FACE_DETECTION_METHOD = "none"


class ClassroomEngagementAnalyzer:
    """Analyzes classroom engagement from video using multiple face detection backends."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.GRID_ROWS = 4
        self.GRID_COLS = 6
        self.WINDOW_SECONDS = 6

        # Initialize face detector based on available method
        if FACE_DETECTION_METHOD == "opencv":
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def get_zone_id(self, x: float, y: float, frame_width: int, frame_height: int) -> str:
        """Convert x,y coordinates to zone ID (R1C1 format)."""
        col = int((x / frame_width) * self.GRID_COLS)
        row = int((y / frame_height) * self.GRID_ROWS)

        # Clamp to valid range
        col = max(0, min(col, self.GRID_COLS - 1))
        row = max(0, min(row, self.GRID_ROWS - 1))

        return f"R{row + 1}C{col + 1}"

    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        faces = []
        if results.multi_face_landmarks:
            frame_height, frame_width = frame.shape[:2]

            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Get face bounding box
                x_coords = [lm.x for lm in landmarks]
                y_coords = [lm.y for lm in landmarks]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Face center
                face_x = int((x_min + x_max) / 2 * frame_width)
                face_y = int((y_min + y_max) / 2 * frame_height)

                # Estimate gaze direction from landmarks
                gaze = self.estimate_gaze_mediapipe(landmarks, frame_width, frame_height)

                faces.append({
                    'x': face_x,
                    'y': face_y,
                    'gaze': gaze,
                    'confidence': 1.0
                })

        return faces

    def detect_faces_face_recognition(self, frame):
        """Detect faces using face_recognition library."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Use HOG for speed

        faces = []
        for (top, right, bottom, left) in face_locations:
            # Calculate face center
            face_x = int((left + right) / 2)
            face_y = int((top + bottom) / 2)

            # Simple gaze estimation (random for face_recognition)
            gaze = self.estimate_gaze_simple(face_x, face_y, frame.shape[1], frame.shape[0])

            faces.append({
                'x': face_x,
                'y': face_y,
                'gaze': gaze,
                'confidence': 0.8
            })

        return faces

    def detect_faces_opencv(self, frame):
        """Detect faces using OpenCV Haar Cascades."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        faces = []
        for (x, y, w, h) in faces_rects:
            # Face center
            face_x = int(x + w/2)
            face_y = int(y + h/2)

            # Simple gaze estimation
            gaze = self.estimate_gaze_simple(face_x, face_y, frame.shape[1], frame.shape[0])

            faces.append({
                'x': face_x,
                'y': face_y,
                'gaze': gaze,
                'confidence': 0.7
            })

        return faces

    def detect_faces(self, frame):
        """Detect faces using the best available method."""
        if FACE_DETECTION_METHOD == "mediapipe":
            return self.detect_faces_mediapipe(frame)
        elif FACE_DETECTION_METHOD == "face_recognition":
            return self.detect_faces_face_recognition(frame)
        elif FACE_DETECTION_METHOD == "opencv":
            return self.detect_faces_opencv(frame)
        else:
            # No face detection available - simulate some faces for demo
            return self.simulate_faces(frame)

    def simulate_faces(self, frame):
        """Simulate face detection for demo purposes when no detector available."""
        height, width = frame.shape[:2]

        # Simulate 5-15 randomly distributed faces
        num_faces = np.random.randint(5, 16)
        faces = []

        for i in range(num_faces):
            face_x = np.random.randint(width // 10, 9 * width // 10)
            face_y = np.random.randint(height // 10, 9 * height // 10)
            gaze = np.random.choice(['forward', 'down', 'left', 'right'], p=[0.5, 0.3, 0.1, 0.1])

            faces.append({
                'x': face_x,
                'y': face_y,
                'gaze': gaze,
                'confidence': 0.5
            })

        return faces

    def estimate_gaze_mediapipe(self, landmarks, frame_width: int, frame_height: int) -> str:
        """Estimate gaze direction from MediaPipe facial landmarks."""
        try:
            # Key landmarks indices
            nose_tip = 1
            left_eye = 33
            right_eye = 263

            nose = landmarks[nose_tip]
            left = landmarks[left_eye]
            right = landmarks[right_eye]

            # Calculate face center
            face_center_x = (left.x + right.x) / 2
            face_center_y = (left.y + right.y) / 2

            # Horizontal and vertical offsets
            horizontal_offset = nose.x - face_center_x
            vertical_offset = nose.y - face_center_y

            # Thresholds
            h_threshold = 0.03
            v_threshold = 0.04

            if abs(vertical_offset) > v_threshold:
                return "down" if vertical_offset > 0 else "forward"
            elif abs(horizontal_offset) > h_threshold:
                return "left" if horizontal_offset < 0 else "right"
            else:
                return "forward"

        except Exception:
            return "forward"

    def estimate_gaze_simple(self, face_x: int, face_y: int, frame_width: int, frame_height: int) -> str:
        """Simple gaze estimation based on face position in frame."""
        # Center-based estimation
        center_x = frame_width / 2
        center_y = frame_height / 2

        # Calculate relative position
        rel_x = (face_x - center_x) / center_x  # -1 to 1
        rel_y = (face_y - center_y) / center_y  # -1 to 1

        # Simple rules
        if abs(rel_y) > 0.3 and rel_y > 0:
            return "down"
        elif abs(rel_x) > 0.4:
            return "left" if rel_x < 0 else "right"
        else:
            return "forward"

    def calculate_engagement_score(self, gaze_direction: str, confidence: float) -> float:
        """Calculate engagement score based on gaze and detection confidence."""
        gaze_weights = {
            "forward": 1.0,
            "down": 0.4,
            "left": 0.6,
            "right": 0.6
        }

        gaze_score = gaze_weights.get(gaze_direction, 0.5)

        # Combine gaze and confidence
        engagement = (gaze_score * 0.8 + confidence * 0.2) * 100

        return min(100, max(0, engagement))

    def analyze_video(self) -> Dict:
        """Main analysis function - returns engagement data structure."""
        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_per_window = fps * self.WINDOW_SECONDS

        print(f"Video stats: {fps} FPS, {total_frames} frames, {frame_width}x{frame_height}")
        print(f"Face detection method: {FACE_DETECTION_METHOD}")
        print(f"Processing {frames_per_window} frames per {self.WINDOW_SECONDS}s window...")

        all_windows = []
        current_window = {
            'window_id': 1,
            'frame_data': [],
            'zone_faces': {},
            'gaze_counts': {'forward': 0, 'down': 0, 'left': 0, 'right': 0},
            'frame_for_thumbnail': None
        }

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            faces = self.detect_faces(frame)

            frame_faces = []

            for face in faces:
                # Get zone
                zone_id = self.get_zone_id(face['x'], face['y'], frame_width, frame_height)

                # Calculate engagement
                engagement = self.calculate_engagement_score(face['gaze'], face['confidence'])

                frame_faces.append({
                    'zone': zone_id,
                    'gaze': face['gaze'],
                    'engagement': engagement
                })

                # Update window data
                if zone_id not in current_window['zone_faces']:
                    current_window['zone_faces'][zone_id] = []
                current_window['zone_faces'][zone_id].append(engagement)
                current_window['gaze_counts'][face['gaze']] += 1

            current_window['frame_data'].append(frame_faces)

            # Store middle frame for thumbnail
            if frame_idx % frames_per_window == frames_per_window // 2:
                current_window['frame_for_thumbnail'] = frame.copy()

            frame_idx += 1

            # Check if window is complete
            if frame_idx % frames_per_window == 0:
                window_data = self._process_window(current_window, frame_idx // frames_per_window, fps)
                all_windows.append(window_data)

                # Reset for next window
                current_window = {
                    'window_id': len(all_windows) + 1,
                    'frame_data': [],
                    'zone_faces': {},
                    'gaze_counts': {'forward': 0, 'down': 0, 'left': 0, 'right': 0},
                    'frame_for_thumbnail': None
                }

            # Progress indicator
            if frame_idx % (fps * 10) == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")

        # Process last incomplete window if exists
        if current_window['frame_data']:
            window_data = self._process_window(current_window, len(all_windows) + 1, fps)
            all_windows.append(window_data)

        cap.release()

        # Identify worst 3 windows
        sorted_windows = sorted(all_windows, key=lambda w: w['group_score'])
        worst_windows = sorted_windows[:3]

        print(f"\nAnalysis complete! Processed {len(all_windows)} windows.")
        avg_engagement = np.mean([w['group_score'] for w in all_windows]) if all_windows else 0
        print(f"Average engagement: {avg_engagement:.1f}")
        print(f"Detection method used: {FACE_DETECTION_METHOD}")

        return {
            'all_windows': all_windows,
            'worst_windows': worst_windows,
            'video_stats': {
                'fps': fps,
                'total_frames': total_frames,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'duration_seconds': total_frames / fps,
                'detection_method': FACE_DETECTION_METHOD
            }
        }

    def _process_window(self, window: Dict, window_id: int, fps: int) -> Dict:
        """Process accumulated window data into structured output."""
        # Calculate zone engagement scores (average per zone)
        zone_engagement = []
        for row in range(1, self.GRID_ROWS + 1):
            for col in range(1, self.GRID_COLS + 1):
                zone_id = f"R{row}C{col}"
                if zone_id in window['zone_faces'] and window['zone_faces'][zone_id]:
                    avg_score = np.mean(window['zone_faces'][zone_id])
                else:
                    avg_score = 0.0

                zone_engagement.append({
                    'zone_id': zone_id,
                    'engagement_score': round(float(avg_score), 2)
                })

        # Calculate group score (overall average)
        all_scores = [z['engagement_score'] for z in zone_engagement if z['engagement_score'] > 0]
        group_score = np.mean(all_scores) if all_scores else 0.0

        # Calculate gaze distribution
        total_gaze = sum(window['gaze_counts'].values())
        gaze_distribution = {}
        if total_gaze > 0:
            for direction, count in window['gaze_counts'].items():
                gaze_distribution[direction] = round((count / total_gaze) * 100, 2)
        else:
            gaze_distribution = {'forward': 0, 'down': 0, 'left': 0, 'right': 0}

        # Calculate timestamp
        timestamp_seconds = (window_id - 1) * self.WINDOW_SECONDS

        return {
            'window_id': window_id,
            'timestamp': timestamp_seconds,
            'group_score': round(float(group_score), 2),
            'zone_engagement': zone_engagement,
            'dominant_gaze_distribution': gaze_distribution,
            'thumbnail_frame': window['frame_for_thumbnail']
        }


def analyze_video(video_path: str) -> Dict:
    """
    Analyze classroom engagement from video.

    Args:
        video_path: Path to the input video file

    Returns:
        Dictionary containing analysis results
    """
    analyzer = ClassroomEngagementAnalyzer(video_path)
    return analyzer.analyze_video()


def create_dashboard(analysis_data: Dict, output_path: str = "engagement_report.html"):
    """
    Generate HTML dashboard with visualizations.

    Args:
        analysis_data: Output from analyze_video()
        output_path: Path for the output HTML file
    """
    all_windows = analysis_data['all_windows']
    worst_windows = analysis_data['worst_windows']
    detection_method = analysis_data['video_stats'].get('detection_method', 'unknown')

    # Prepare data for charts
    timestamps = [w['timestamp'] for w in all_windows]
    group_scores = [w['group_score'] for w in all_windows]

    # Calculate average zone engagement across all windows
    zone_heatmap_data = {}
    for window in all_windows:
        for zone_data in window['zone_engagement']:
            zone_id = zone_data['zone_id']
            if zone_id not in zone_heatmap_data:
                zone_heatmap_data[zone_id] = []
            zone_heatmap_data[zone_id].append(zone_data['engagement_score'])

    # Average per zone
    zone_avg = {zone: np.mean(scores) for zone, scores in zone_heatmap_data.items()}

    # Calculate overall gaze distribution
    total_gaze = {'forward': 0, 'down': 0, 'left': 0, 'right': 0}
    for window in all_windows:
        for direction in total_gaze.keys():
            total_gaze[direction] += window['dominant_gaze_distribution'].get(direction, 0)

    # Normalize
    total = sum(total_gaze.values())
    if total > 0:
        overall_gaze = {k: (v / total) * 100 for k, v in total_gaze.items()}
    else:
        overall_gaze = total_gaze

    # Convert worst window thumbnails to base64
    worst_thumbnails = []
    for w in worst_windows:
        if w['thumbnail_frame'] is not None:
            # Resize for display
            thumb = cv2.resize(w['thumbnail_frame'], (320, 240))
            _, buffer = cv2.imencode('.jpg', thumb)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            worst_thumbnails.append({
                'timestamp': w['timestamp'],
                'score': w['group_score'],
                'image': img_base64
            })

    # Generate worst windows HTML
    worst_windows_html = _generate_worst_windows_html(worst_thumbnails)

    # Convert data to JSON strings for JavaScript
    timestamps_json = json.dumps(timestamps)
    group_scores_json = json.dumps(group_scores)
    zone_avg_json = json.dumps(zone_avg)
    gaze_forward = overall_gaze['forward']
    gaze_down = overall_gaze['down']
    gaze_left = overall_gaze['left']
    gaze_right = overall_gaze['right']

    # Build HTML (NOT using f-string for JavaScript parts)
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classroom Engagement Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .detection-info {
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 10px;
            margin-top: 15px;
            font-size: 0.9em;
        }

        .content {
            padding: 40px;
        }

        .section {
            margin-bottom: 50px;
        }

        .section h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 25px;
            font-size: 1.8em;
        }

        .chart-container {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        canvas {
            max-width: 100%;
            height: auto !important;
        }

        .heatmap-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            grid-template-rows: repeat(4, 100px);
            gap: 10px;
            margin: 20px 0;
        }

        .zone-cell {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            color: white;
            font-weight: bold;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }

        .zone-cell:hover {
            transform: scale(1.05);
        }

        .zone-label {
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .zone-score {
            font-size: 1.5em;
        }

        .worst-windows {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }

        .window-card {
            background: #f8f9fa;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }

        .window-card:hover {
            transform: translateY(-5px);
        }

        .window-card img {
            width: 100%;
            height: auto;
            display: block;
        }

        .window-info {
            padding: 20px;
        }

        .window-info h3 {
            color: #667eea;
            margin-bottom: 10px;
        }

        .window-info p {
            color: #666;
            line-height: 1.6;
        }

        .score-badge {
            display: inline-block;
            background: #dc3545;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }

        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .legend-color {
            width: 30px;
            height: 30px;
            border-radius: 5px;
        }

        @media (max-width: 768px) {
            .heatmap-grid {
                grid-template-columns: repeat(3, 1fr);
                grid-template-rows: repeat(8, 80px);
            }

            .header h1 {
                font-size: 1.8em;
            }
        }
    </style>
    <script>
        const Chart = (function() {
            function Chart(ctx, config) {
                this.ctx = ctx;
                this.config = config;
                this.draw();
            }

            Chart.prototype.draw = function() {
                const canvas = this.ctx.canvas;
                const config = this.config;
                const ctx = this.ctx;

                if (config.type === 'line') {
                    this.drawLineChart(canvas, ctx, config.data, config.options);
                } else if (config.type === 'bar') {
                    this.drawBarChart(canvas, ctx, config.data, config.options);
                }
            };

            Chart.prototype.drawLineChart = function(canvas, ctx, data, options) {
                const padding = 50;
                const width = canvas.width - 2 * padding;
                const height = canvas.height - 2 * padding;

                ctx.clearRect(0, 0, canvas.width, canvas.height);

                ctx.strokeStyle = '#666';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(padding, padding);
                ctx.lineTo(padding, canvas.height - padding);
                ctx.lineTo(canvas.width - padding, canvas.height - padding);
                ctx.stroke();

                const labels = data.labels;
                const values = data.datasets[0].data;
                const max = Math.max(...values);
                const min = Math.min(...values);
                const range = max - min || 1;

                ctx.strokeStyle = data.datasets[0].borderColor || '#667eea';
                ctx.lineWidth = 3;
                ctx.beginPath();

                for (let i = 0; i < values.length; i++) {
                    const x = padding + (i / (values.length - 1)) * width;
                    const y = canvas.height - padding - ((values[i] - min) / range) * height;

                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }

                ctx.stroke();

                ctx.fillStyle = data.datasets[0].borderColor || '#667eea';
                for (let i = 0; i < values.length; i++) {
                    const x = padding + (i / (values.length - 1)) * width;
                    const y = canvas.height - padding - ((values[i] - min) / range) * height;

                    ctx.beginPath();
                    ctx.arc(x, y, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }

                ctx.fillStyle = '#666';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';

                const labelSkip = Math.ceil(labels.length / 10);
                for (let i = 0; i < labels.length; i += labelSkip) {
                    const x = padding + (i / (values.length - 1)) * width;
                    ctx.fillText(labels[i] + 's', x, canvas.height - padding + 20);
                }

                ctx.textAlign = 'right';
                for (let i = 0; i <= 5; i++) {
                    const y = canvas.height - padding - (i / 5) * height;
                    const value = min + (i / 5) * range;
                    ctx.fillText(value.toFixed(0), padding - 10, y + 5);
                }

                ctx.font = 'bold 16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(options.plugins.title.text, canvas.width / 2, 25);
            };

            Chart.prototype.drawBarChart = function(canvas, ctx, data, options) {
                const padding = 60;
                const width = canvas.width - 2 * padding;
                const height = canvas.height - 2 * padding;

                ctx.clearRect(0, 0, canvas.width, canvas.height);

                const labels = data.labels;
                const values = data.datasets[0].data;
                const max = Math.max(...values) || 100;

                const barHeight = height / labels.length - 10;

                ctx.font = 'bold 16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(options.plugins.title.text, canvas.width / 2, 25);

                for (let i = 0; i < labels.length; i++) {
                    const y = padding + i * (height / labels.length);
                    const barWidth = (values[i] / max) * width;

                    ctx.fillStyle = data.datasets[0].backgroundColor[i] || '#667eea';
                    ctx.fillRect(padding, y, barWidth, barHeight);

                    ctx.fillStyle = '#333';
                    ctx.font = '14px Arial';
                    ctx.textAlign = 'right';
                    ctx.fillText(labels[i], padding - 10, y + barHeight / 2 + 5);

                    ctx.textAlign = 'left';
                    ctx.fillText(values[i].toFixed(1) + '%', padding + barWidth + 10, y + barHeight / 2 + 5);
                }
            };

            return Chart;
        })();
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Classroom Engagement Analysis</h1>
            <p>AI-Powered Engagement Monitoring Dashboard</p>
            <div class="detection-info">
                🔍 Face Detection Method: """ + detection_method.title() + """
            </div>
        </div>

        <div class="content">
            <!-- Engagement Over Time -->
            <div class="section">
                <h2>📈 Engagement Score Over Time</h2>
                <div class="chart-container">
                    <canvas id="engagementChart" width="800" height="400"></canvas>
                </div>
            </div>

            <!-- Spatial Heatmap -->
            <div class="section">
                <h2>🗺️ Spatial Engagement Heatmap</h2>
                <div class="chart-container">
                    <div class="heatmap-grid" id="heatmapGrid"></div>
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background: #dc3545;"></div>
                            <span>Low (&lt;40)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ffc107;"></div>
                            <span>Medium (40-70)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #28a745;"></div>
                            <span>High (&gt;70)</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Worst 3 Windows -->
            <div class="section">
                <h2>⚠️ Lowest Engagement Moments (Evidence)</h2>
                <div class="worst-windows">
                    """ + worst_windows_html + """
                </div>
            </div>

            <!-- Gaze Distribution -->
            <div class="section">
                <h2>👁️ Gaze Direction Distribution</h2>
                <div class="chart-container">
                    <canvas id="gazeChart" width="600" height="400"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        const engagementCtx = document.getElementById('engagementChart').getContext('2d');
        new Chart(engagementCtx, {
            type: 'line',
            data: {
                labels: """ + timestamps_json + """,
                datasets: [{
                    label: 'Group Engagement Score',
                    data: """ + group_scores_json + """,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Group Engagement Score (every 6 seconds)'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Engagement Score'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    }
                }
            }
        });

        const heatmapGrid = document.getElementById('heatmapGrid');
        const zoneData = """ + zone_avg_json + """;

        for (let row = 1; row <= 4; row++) {
            for (let col = 1; col <= 6; col++) {
                const zoneId = 'R' + row + 'C' + col;
                const score = zoneData[zoneId] || 0;

                const cell = document.createElement('div');
                cell.className = 'zone-cell';

                let color;
                if (score < 40) {
                    color = '#dc3545';
                } else if (score < 70) {
                    color = '#ffc107';
                } else {
                    color = '#28a745';
                }

                cell.style.backgroundColor = color;
                cell.innerHTML = '<div class="zone-label">' + zoneId + '</div><div class="zone-score">' + score.toFixed(1) + '</div>';

                heatmapGrid.appendChild(cell);
            }
        }

        const gazeCtx = document.getElementById('gazeChart').getContext('2d');
        new Chart(gazeCtx, {
            type: 'bar',
            data: {
                labels: ['Forward', 'Down', 'Left', 'Right'],
                datasets: [{
                    label: 'Percentage',
                    data: [
                        """ + str(gaze_forward) + """,
                        """ + str(gaze_down) + """,
                        """ + str(gaze_left) + """,
                        """ + str(gaze_right) + """
                    ],
                    backgroundColor: [
                        '#28a745',
                        '#ffc107',
                        '#17a2b8',
                        '#6f42c1'
                    ]
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Overall Gaze Direction Distribution'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Percentage (%)'
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Dashboard created: {output_path}")
    print(f"Face detection method used: {detection_method}")


def _generate_worst_windows_html(thumbnails: List[Dict]) -> str:
    """Generate HTML for worst windows evidence section."""
    html_parts = []

    for i, thumb in enumerate(thumbnails, 1):
        minutes = thumb['timestamp'] // 60
        seconds = thumb['timestamp'] % 60
        time_str = f"{int(minutes)}:{int(seconds):02d}"

        html_parts.append(f"""
        <div class="window-card">
            <img src="data:image/jpeg;base64,{thumb['image']}" alt="Window {i}">
            <div class="window-info">
                <h3>Window #{i}</h3>
                <p><strong>Timestamp:</strong> {time_str}</p>
                <p><strong>Engagement Score:</strong> <span class="score-badge">{thumb['score']:.1f}</span></p>
            </div>
        </div>
        """)

    return ''.join(html_parts)


def save_integration_json(analysis_data: Dict, output_path: str = "engagement_output.json"):
    """
    Save engagement data in integration JSON format.

    Args:
        analysis_data: Output from analyze_video()
        output_path: Path for the output JSON file
    """
    all_windows = analysis_data['all_windows']
    detection_method = analysis_data['video_stats'].get('detection_method', 'unknown')

    # Format for integration
    output_data = {
        "session_id": "classroom_001",
        "total_windows": len(all_windows),
        "metadata": {
            "detection_method": detection_method,
            "analysis_timestamp": str(np.datetime64('now')),
            "author": "Harshavardhan Perla (25CS60R72)"
        },
        "windows": []
    }

    for window in all_windows:
        output_data["windows"].append({
            "window_id": window['window_id'],
            "timestamp": window['timestamp'],
            "group_score": window['group_score'],
            "zone_engagement": window['zone_engagement'],
            "dominant_gaze_distribution": window['dominant_gaze_distribution']
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"Integration JSON saved: {output_path}")
    print(f"Face detection method: {detection_method}")


def main():
    """Main execution function."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python solution.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print("=" * 60)
    print("CLASSROOM ENGAGEMENT ANALYSIS - ROBUST VERSION")
    print("=" * 60)
    print(f"Processing video: {video_path}\n")

    # Step 1: Analyze video
    print("Step 1/3: Analyzing video...")
    analysis_data = analyze_video(video_path)

    # Step 2: Create dashboard
    print("\nStep 2/3: Creating dashboard...")
    create_dashboard(analysis_data)

    # Step 3: Save JSON
    print("\nStep 3/3: Saving integration JSON...")
    save_integration_json(analysis_data)

    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  📊 engagement_report.html - Interactive dashboard")
    print("  📄 engagement_output.json - Integration data")
    print(f"\nFace detection method used: {FACE_DETECTION_METHOD}")
    print("\nOpen engagement_report.html in your browser to view the results.")


if __name__ == "__main__":
    main()