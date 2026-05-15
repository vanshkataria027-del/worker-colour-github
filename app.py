import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from styles import apply_styles

# Page config
st.set_page_config(
    page_title="Worker Uniform Colour Detector",
    page_icon="👷",
    layout="wide"
)

apply_styles()

# Header
st.markdown('<h1 class="main-title">👷 Worker Uniform Colour Detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload a video to detect whether workers are wearing '
    'their <b>blue uniforms</b> in real-time using YOLOv8 + OpenCV.</p>',
    unsafe_allow_html=True,
)

st.divider()

# Load YOLO model (cached)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported: MP4, AVI, MOV, MKV"
    )

    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    blue_pixel_threshold = st.slider("Blue Pixel Threshold", 1000, 20000, 5000, 500,
                                      help="Minimum blue pixels to count as uniform")
    frame_skip = st.slider("Process every N frames", 1, 10, 2,
                            help="Higher = faster but less smooth")

with col2:
    st.markdown("### 📊 Detection Results")

    if uploaded_file:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        video_path = tfile.name

        if st.button("▶️ Start Detection", type="primary"):

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            st.markdown(f"**Total Frames:** {total_frames} | **FPS:** {fps:.1f}")

            # Placeholders
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            progress_bar = st.progress(0)

            frame_idx = 0
            total_blue = 0
            total_no_blue = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Skip frames for speed
                if frame_idx % frame_skip != 0:
                    continue

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                results = model(frame, classes=[0], verbose=False)[0]

                blue_count = 0
                no_blue_count = 0

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    if conf < confidence_threshold:
                        continue

                    person_crop = hsv[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue

                    lower_blue = np.array([90, 50, 50])
                    upper_blue = np.array([130, 255, 255])
                    mask = cv2.inRange(person_crop, lower_blue, upper_blue)

                    if mask.sum() > blue_pixel_threshold:
                        blue_count += 1
                        color = (255, 0, 0)
                        label = f"Blue Uniform {conf:.2f}"
                    else:
                        no_blue_count += 1
                        color = (0, 0, 255)
                        label = f"No Uniform {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                cv2.putText(frame, f"Blue: {blue_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"No Blue: {no_blue_count}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                total_blue += blue_count
                total_no_blue += no_blue_count

                # Show frame in Streamlit (BGR -> RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Update stats
                stats_placeholder.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | 🟦 Blue Uniform Detections | `{total_blue}` |
                | 🟥 No Uniform Detections | `{total_no_blue}` |
                | 📽️ Frame | `{frame_idx} / {total_frames}` |
                """)

                # Progress
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            os.unlink(video_path)
            st.success("✅ Detection complete!")

    else:
        st.info("👈 Upload a video to get started.")

st.divider()
st.markdown('<p class="footer">Worker Uniform Colour Detector • YOLOv8 + OpenCV + Streamlit</p>', unsafe_allow_html=True)
