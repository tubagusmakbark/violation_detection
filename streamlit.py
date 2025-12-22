import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd
import time
import zipfile
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Violation Detection", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'detection_started' not in st.session_state:
    st.session_state['detection_started'] = False
if 'violation_list' not in st.session_state:
    st.session_state['violation_list'] = []

st.title("🚦 AI Lane Violation Detection System")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Settings")

# Model Selection
model_path = "best.pt" 

# Detection Settings
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
class_id = st.sidebar.number_input("Class ID (1=Motorcycle)", value=1, step=1)

# Size Filter
st.sidebar.subheader("Size Filter")
min_w = st.sidebar.number_input("Min Width (px)", value=100)
min_h = st.sidebar.number_input("Min Height (px)", value=100)

# Lane Position
st.sidebar.subheader("Lane Position Setup")
lane_x1 = st.sidebar.slider("Left Line X", 0, 1280, 400)
lane_x2 = st.sidebar.slider("Right Line X", 0, 1280, 800)
lane_y_top = st.sidebar.slider("Top Boundary Y", 0, 720, 200)
lane_y_bot = st.sidebar.slider("Bottom Boundary Y", 0, 720, 700) 

points = [[lane_x1, lane_y_top], [lane_x1, lane_y_bot], [lane_x2, lane_y_top], [lane_x2, lane_y_bot]]

# --- DOWNLOAD SECTION ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Reports & Evidence")

if st.session_state['violation_list']:
    # 1. CSV Download
    df = pd.DataFrame(st.session_state['violation_list'])
    # Drop the image bytes column for the CSV report to keep it clean
    df_csv = df.drop(columns=['ImageBytes'])
    csv = df_csv.to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        label="📄 Download Report (CSV)",
        data=csv,
        file_name="violation_report.csv",
        mime="text/csv",
    )

    # 2. ZIP Images Download
    # Create an in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for item in st.session_state['violation_list']:
            file_name = f"violation_id_{item['ID']}.jpg"
            zf.writestr(file_name, item['ImageBytes'])
    
    st.sidebar.download_button(
        label="📸 Download Images (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="violation_evidence.zip",
        mime="application/zip"
    )

else:
    st.sidebar.info("No violations yet.")

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
    upscaled = cv2.resize(image, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    contrast = cv2.convertScaleAbs(upscaled, alpha=1.3, beta=10)
    return contrast

# --- MAIN APP LOGIC ---
model = load_model(model_path)

video_source = st.radio("Select Video Source", ["Upload Video", "Use Webcam"], horizontal=True)
input_path = None

if video_source == "Upload Video":
    video_file = st.file_uploader("Upload MP4/AVI", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        input_path = tfile.name
else:
    input_path = 0 

if input_path is not None:
    cap = cv2.VideoCapture(input_path)
    
    # --- PHASE 1: SETUP ---
    if not st.session_state['detection_started']:
        st.info("👆 Adjust sliders. Click 'Start Detection' when ready.")
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            poly_pts = np.array([points[0], points[1], points[3], points[2]], np.int32)
            cv2.polylines(frame, [poly_pts], True, (0, 255, 255), 3)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly_pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Fix Color for Setup Preview too
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Setup Preview", use_container_width=True)
            
            if st.button("🚀 Confirm Lane & Start Detection", type="primary"):
                st.session_state['detection_started'] = True
                st.rerun() 
        else:
            st.error("Could not read video.")

    # --- PHASE 2: DETECTION ---
    else:
        if st.button("⏹️ Stop / Reset"):
            st.session_state['detection_started'] = False
            st.session_state['violation_list'] = [] 
            st.rerun()

        st_frame = st.empty()
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Live Violations")
            violation_log = st.empty()

        captured_ids = set()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video finished.")
                break
                
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            
            results = model.track(frame, persist=True, conf=confidence, classes=[class_id], verbose=False)
            
            poly_pts = np.array([points[0], points[1], points[3], points[2]], np.int32)
            cv2.polylines(frame, [poly_pts], True, (0, 255, 255), 2)
            
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
                                captured_ids.add(track_id)
                                
                                # 1. Crop & Enhance (BGR format)
                                crop_bgr = frame[y1:y2, x1:x2]
                                enhanced_bgr = enhance_image(crop_bgr)
                                
                                # 2. Convert to RGB for Streamlit Display
                                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                                
                                with col2:
                                    st.image(enhanced_rgb, caption=f"Violation ID: {int(track_id)}", width=150)
                                
                                # 3. Save BGR bytes for ZIP Download (Standard Format)
                                success, encoded_jpg = cv2.imencode('.jpg', enhanced_bgr)
                                
                                timestamp = time.strftime("%H:%M:%S")
                                st.session_state['violation_list'].append({
                                    "ID": int(track_id),
                                    "Time": timestamp,
                                    "Status": "Violation",
                                    "Width": obj_w,
                                    "Height": obj_h,
                                    "ImageBytes": encoded_jpg.tobytes() # Store binary for ZIP
                                })
                                label = "CAPTURED"
                            else:
                                label = "WAITING..."
                        else:
                            label = "DONE"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Fix Main Video Color too
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()

elif video_source == "Use Webcam":
    st.warning("Ensure webcam is not being used by another app.")