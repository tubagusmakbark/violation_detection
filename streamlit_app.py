import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Violation Detection", layout="wide")

st.title("🚦 AI Lane Violation Detection System")
st.markdown("Upload a video or use a sample to detect lane violations using YOLOv8.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Settings")

# Model Selection
model_source = st.sidebar.radio("Model Source", ["Use Standard YOLOv8n", "Upload Custom Model (.pt)"])
model_path = "yolov8n.pt" # Default

if model_source == "Upload Custom Model (.pt)":
    uploaded_model = st.sidebar.file_uploader("Upload .pt file", type=["pt"])
    if uploaded_model:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(uploaded_model.read())
            model_path = tmp.name
        st.sidebar.success("Custom model loaded!")

# Detection Settings
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
class_id = st.sidebar.number_input("Class ID (1=Bike, 2=Car, 3=Moto)", value=3, step=1)

# Size Filter
st.sidebar.subheader("Size Filter")
min_w = st.sidebar.number_input("Min Width (px)", value=100)
min_h = st.sidebar.number_input("Min Height (px)", value=100)

# Lane Line Adjustment (Sliders instead of Mouse Drag)
st.sidebar.subheader("Lane Position")
lane_x1 = st.sidebar.slider("Left Line X", 0, 1280, 400)
lane_x2 = st.sidebar.slider("Right Line X", 0, 1280, 800)
lane_y_top = 200
lane_y_bot = 600

# Define points based on sliders
points = [[lane_x1, lane_y_top], [lane_x1, lane_y_bot], [lane_x2, lane_y_top], [lane_x2, lane_y_bot]]

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model(path):
    return YOLO(path)

def is_inside_lane(center_point, polygon_points):
    poly = np.array([
        polygon_points[0], polygon_points[1], 
        polygon_points[3], polygon_points[2] 
    ], np.int32)
    return cv2.pointPolygonTest(poly, center_point, False) >= 0

def enhance_image(image):
    if image is None or image.size == 0: return image
    h, w = image.shape[:2]
    # Upscale and sharpen
    upscaled = cv2.resize(image, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    contrast = cv2.convertScaleAbs(upscaled, alpha=1.3, beta=10)
    return contrast

# --- MAIN APP ---
model = load_model(model_path)

# Video Input Source
video_source = st.radio("Select Video Source", ["Upload Video", "Use Webcam"])
input_path = None

if video_source == "Upload Video":
    video_file = st.file_uploader("Upload MP4/AVI", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        input_path = tfile.name
else:
    input_path = 0 # Webcam ID

# Start Button
if input_path is not None and st.button("🚀 Start Detection"):
    
    cap = cv2.VideoCapture(input_path)
    
    # Create placeholders for the video and the metrics
    st_frame = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Live Feed")
    with col2:
        st.subheader("Recent Violations")
        violation_placeholder = st.empty()

    captured_ids = set()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video finished.")
            break
            
        # Resize for 4K handling (Standard HD for Web)
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        
        # 1. Run Tracking
        results = model.track(frame, persist=True, conf=confidence, classes=[class_id], verbose=False)
        
        # 2. Draw Lane
        poly_pts = np.array([points[0], points[1], points[3], points[2]], np.int32)
        cv2.polylines(frame, [poly_pts], True, (0, 255, 255), 2)
        
        # Transparent Overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly_pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # 3. Process Detections
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), y2
                obj_w, obj_h = x2 - x1, y2 - y1
                
                color = (0, 255, 0)
                label = f"ID {int(track_id)}"
                
                if is_inside_lane((cx, cy), points):
                    color = (0, 0, 255)
                    
                    if track_id not in captured_ids:
                        if obj_w >= min_w and obj_h >= min_h:
                            # Capture Violation
                            crop = frame[y1:y2, x1:x2]
                            enhanced = enhance_image(crop)
                            
                            # Show in the "Recent Violations" column
                            violation_placeholder.image(enhanced, caption=f"VIOLATION ID: {int(track_id)}", width=200)
                            captured_ids.add(track_id)
                            label = "CAPTURED"
                        else:
                            label = "WAITING..."
                    else:
                        label = "DONE"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 4. Display in Streamlit
        # Convert BGR to RGB for browser display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()