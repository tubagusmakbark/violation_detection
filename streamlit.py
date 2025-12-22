import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd
import time
import zipfile
import io
from PIL import Image

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

model_path = "best.pt" 

# Performance & Detection
frame_skip = st.sidebar.slider("Frame Skip (Video Only)", 1, 5, 3)
resolution_scale = st.sidebar.radio("Processing Resolution", ["480p (Fast)", "720p (HD)"], index=0)

if resolution_scale == "720p (HD)":
    TARGET_W, TARGET_H = 1280, 720
else:
    TARGET_W, TARGET_H = 854, 480

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
class_id = st.sidebar.number_input("Class ID (1=Motorcycle)", value=1, step=1)

st.sidebar.subheader("Size Filter")
min_w = st.sidebar.number_input("Min Width (px)", value=50)
min_h = st.sidebar.number_input("Min Height (px)", value=50)

st.sidebar.subheader("Lane Position Setup")
lane_x1 = st.sidebar.slider("Left Line X", 0, TARGET_W, int(TARGET_W * 0.3))
lane_x2 = st.sidebar.slider("Right Line X", 0, TARGET_W, int(TARGET_W * 0.6))
lane_y_top = st.sidebar.slider("Top Boundary Y", 0, TARGET_H, int(TARGET_H * 0.3))
lane_y_bot = st.sidebar.slider("Bottom Boundary Y", 0, TARGET_H, int(TARGET_H * 0.9))

points = [[lane_x1, lane_y_top], [lane_x1, lane_y_bot], [lane_x2, lane_y_top], [lane_x2, lane_y_bot]]

# --- DOWNLOAD SECTION ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Reports & Evidence")

if st.session_state['violation_list']:
    df = pd.DataFrame(st.session_state['violation_list'])
    df_csv = df.drop(columns=['ImageBytes']) 
    csv = df_csv.to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        label="📄 Download Report (CSV)",
        data=csv,
        file_name="violation_report.csv",
        mime="text/csv",
    )

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
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure 'best.pt' is in the folder.")
        return None

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

def process_frame(frame, model, points, captured_ids, is_video=True):
    """Unified function to process both Video frames and Static Images"""
    # Resize
    frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    
    # Run YOLO (persist=True is better for video, False for single image)
    results = model.track(frame, persist=is_video, conf=confidence, classes=[class_id], verbose=False)
    
    # Draw Lane
    poly_pts = np.array([points[0], points[1], points[3], points[2]], np.int32)
    cv2.polylines(frame, [poly_pts], True, (0, 255, 255), 2)
    
    violation_found = False
    
    if results[0].boxes.id is not None or (not is_video and results[0].boxes.xyxy is not None):
        # Handle cases where ID might be None in single image mode
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # If video, use track IDs. If image, generate fake IDs based on index
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy()
        else:
            track_ids = [i for i in range(len(boxes))]

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), y2
            obj_w, obj_h = x2 - x1, y2 - y1
            
            color = (0, 255, 0)
            label = f"ID {int(track_id)}"
            
            if is_inside_lane((cx, cy), points):
                color = (0, 0, 255)
                
                # Check duplication
                if track_id not in captured_ids:
                    if obj_w >= min_w and obj_h >= min_h:
                        captured_ids.add(track_id)
                        violation_found = True
                        
                        # Capture Logic
                        crop_bgr = frame[y1:y2, x1:x2]
                        enhanced_bgr = enhance_image(crop_bgr)
                        
                        # Save Evidence
                        success, encoded_jpg = cv2.imencode('.jpg', enhanced_bgr)
                        timestamp = time.strftime("%H:%M:%S")
                        
                        st.session_state['violation_list'].append({
                            "ID": int(track_id),
                            "Time": timestamp,
                            "Status": "Violation",
                            "Width": obj_w,
                            "Height": obj_h,
                            "ImageBytes": encoded_jpg.tobytes()
                        })
                        label = "CAPTURED"
                        
                        # Show Sidebar Preview
                        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                        st.sidebar.image(enhanced_rgb, caption=f"Captured ID {int(track_id)}", width=150)
                        
                    else:
                        label = "WAITING..."
                else:
                    label = "DONE"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    return frame, violation_found

# --- MAIN APP LOGIC ---
model = load_model(model_path)

if model:
    # 1. Source Selection (Added 'Upload Image')
    video_source = st.radio("Select Input Source", ["Upload Video", "Upload Image", "Use Webcam"], horizontal=True)
    
    input_source = None
    is_static_image = False

    if video_source == "Upload Video":
        video_file = st.file_uploader("Upload MP4/AVI", type=["mp4", "avi", "mov"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            input_source = tfile.name
            
    elif video_source == "Upload Image":
        image_file = st.file_uploader("Upload JPG/PNG", type=["jpg", "png", "jpeg"])
        if image_file:
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            input_source = cv2.imdecode(file_bytes, 1) # 1 = Color
            is_static_image = True
            
    else:
        input_source = 0 # Webcam

    # --- PROCESSING ---
    if input_source is not None:
        
        # A. STATIC IMAGE LOGIC
        if is_static_image:
            frame = input_source.copy()
            
            # Setup Preview for Image
            st.info("Adjust lines, then click 'Process Image'.")
            
            # Show Setup Overlay
            preview = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            poly_pts = np.array([points[0], points[1], points[3], points[2]], np.int32)
            cv2.polylines(preview, [poly_pts], True, (0, 255, 255), 3)
            
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb, caption="Image Preview", use_container_width=True)
            
            if st.button("📸 Process Image", type="primary"):
                # Run Detection ONCE
                processed_frame, found = process_frame(frame, model, points, set(), is_video=False)
                
                # Show Result
                result_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, caption="Processed Result", use_container_width=True)
                
                if found:
                    st.success("✅ Violation Detected & Saved to Report!")
                else:
                    st.warning("No violations detected in this image.")

        # B. VIDEO/WEBCAM LOGIC
        else:
            cap = cv2.VideoCapture(input_source)
            
            if not st.session_state['detection_started']:
                st.info("👆 Adjust sliders. Click 'Start Detection' when ready.")
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                    poly_pts = np.array([points[0], points[1], points[3], points[2]], np.int32)
                    cv2.polylines(frame, [poly_pts], True, (0, 255, 255), 3)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Video Preview", use_container_width=True)
                    
                    if st.button("🚀 Confirm Lane & Start Detection", type="primary"):
                        st.session_state['detection_started'] = True
                        st.rerun()
            else:
                if st.button("⏹️ Stop / Reset"):
                    st.session_state['detection_started'] = False
                    st.session_state['violation_list'] = [] 
                    st.rerun()

                st_frame = st.empty()
                captured_ids = set()
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue 
                        
                    # Process Frame using the shared function
                    processed_frame, _ = process_frame(frame, model, points, captured_ids, is_video=True)
                    
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                
                cap.release()