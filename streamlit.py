import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd
import time
import zipfile
import io
import yt_dlp

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

# NEW: Aspect Ratio Control
resize_mode = st.sidebar.radio("Aspect Ratio Mode", ["Crop to Fill (No Distortion)", "Stretch (Flattened)"], index=0)

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
    st.sidebar.success(f"{len(st.session_state['violation_list'])} Violations Captured")
    
    # Prepare CSV
    df = pd.DataFrame(st.session_state['violation_list'])
    df_csv = df.drop(columns=['ImageBytes']) 
    csv = df_csv.to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        label="📄 Download Report (CSV)",
        data=csv,
        file_name="violation_report.csv",
        mime="text/csv",
    )

    # Prepare ZIP
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
    
    if st.sidebar.button("🗑️ Clear All Data"):
        st.session_state['violation_list'] = []
        st.rerun()
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

# --- NEW: SMART RESIZE FUNCTION ---
def smart_resize(frame, target_w, target_h, mode="Crop to Fill (No Distortion)"):
    if mode == "Stretch (Flattened)":
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # Mode: Crop to Fill (Aspect Fill)
    h, w = frame.shape[:2]
    img_ratio = w / h
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        # Image is wider than target: Resize by height, crop width
        new_h = target_h
        new_w = int(w * (target_h / h))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center Crop Width
        start_x = (new_w - target_w) // 2
        return resized[:, start_x:start_x+target_w]
    else:
        # Image is taller/same: Resize by width, crop height
        new_w = target_w
        new_h = int(h * (target_w / w))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center Crop Height
        start_y = (new_h - target_h) // 2
        return resized[start_y:start_y+target_h, :]

def process_frame(frame, model, points, captured_ids, is_video=True):
    # Apply Smart Resize instead of Hard Resize
    frame = smart_resize(frame, TARGET_W, TARGET_H, resize_mode)
    
    # Run YOLO
    results = model.track(frame, persist=is_video, conf=confidence, classes=[class_id], verbose=False)
    
    # Draw Lane
    poly_pts = np.array([points[0], points[1], points[3], points[2]], np.int32)
    cv2.polylines(frame, [poly_pts], True, (0, 255, 255), 2)
    
    violation_found = False
    
    if results[0].boxes.id is not None or (not is_video and results[0].boxes.xyxy is not None):
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
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
                
                if track_id not in captured_ids:
                    if obj_w >= min_w and obj_h >= min_h:
                        captured_ids.add(track_id)
                        violation_found = True
                        
                        # Capture Logic
                        crop_bgr = frame[y1:y2, x1:x2]
                        enhanced_bgr = enhance_image(crop_bgr)
                        
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
                    else:
                        label = "WAITING..."
                else:
                    label = "DONE"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    return frame, violation_found

@st.cache_resource
def get_youtube_stream_url(youtube_url):
    ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True, 'no_warnings': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info['url'], info.get('title', 'YouTube Video')
    except Exception as e:
        return None, None

# --- MAIN APP LOGIC ---
model = load_model(model_path)

if model:
    # 1. Source Selection
    video_source = st.radio("Select Input Source", ["Upload Video", "Upload Image", "YouTube URL"], horizontal=True)
    
    input_source = None
    is_static_image = False
    video_title = "Video Stream"

    if video_source == "Upload Video":
        video_file = st.file_uploader("Upload MP4/AVI", type=["mp4", "avi", "mov"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            input_source = tfile.name
            
    elif video_source == "Upload Image":
        image_file = st.file_uploader("Upload JPG/PNG", type=["jpg", "png", "jpeg"])
        if image_file:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            input_source = cv2.imdecode(file_bytes, 1)
            is_static_image = True

    elif video_source == "YouTube URL":
        yt_url = st.text_input("Paste YouTube URL here", placeholder="https://www.youtube.com/watch?v=...")
        if yt_url:
            with st.spinner("Extracting stream URL..."):
                stream_url, title = get_youtube_stream_url(yt_url)
                if stream_url:
                    input_source = stream_url
                    video_title = title
                    st.success(f"Loaded: {title}")
                else:
                    st.error("Could not extract video stream. Check the URL.")

    # --- PROCESSING ---
    if input_source is not None:
        
        # A. STATIC IMAGE
        if is_static_image:
            frame = input_source.copy()
            st.info("Adjust lines, then click 'Process Image'.")
            
            # Smart Resize for Preview
            preview = smart_resize(frame, TARGET_W, TARGET_H, resize_mode)
            
            poly_pts = np.array([points[0], points[1], points[3], points[2]], np.int32)
            cv2.polylines(preview, [poly_pts], True, (0, 255, 255), 3)
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Image Preview", use_container_width=True)
            
            if st.button("📸 Process Image", type="primary"):
                processed_frame, found = process_frame(frame, model, points, set(), is_video=False)
                st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Processed Result", use_container_width=True)
                if found:
                    st.success("✅ Violation Detected & Saved to Report!")
                    st.rerun()
                else:
                    st.warning("No violations detected.")

        # B. VIDEO/YOUTUBE
        else:
            cap = cv2.VideoCapture(input_source)
            
            if not st.session_state['detection_started']:
                # SETUP MODE
                st.info(f"Source: {video_title}. Adjust sliders. Click 'Start Detection' when ready.")
                
                if st.session_state['violation_list']:
                     st.warning(f"⚠️ {len(st.session_state['violation_list'])} violations in memory.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚀 Confirm Lane & Start Detection", type="primary"):
                        st.session_state['detection_started'] = True
                        st.rerun()
                with col2:
                    if st.button("🔄 Reset All Data"):
                        st.session_state['violation_list'] = []
                        st.session_state['detection_started'] = False
                        st.rerun()

                # Preview Frame with Smart Resize
                ret, frame = cap.read()
                if ret:
                    frame = smart_resize(frame, TARGET_W, TARGET_H, resize_mode)
                    
                    poly_pts = np.array([points[0], points[1], points[3], points[2]], np.int32)
                    cv2.polylines(frame, [poly_pts], True, (0, 255, 255), 3)
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Preview", use_container_width=True)
            
            else:
                # RUNNING MODE
                c1, c2 = st.columns(2)
                stop_pressed = c1.button("⏹️ Stop (Keep Data)")
                reset_pressed = c2.button("🔄 Reset (Clear Data)")
                
                if stop_pressed:
                    st.session_state['detection_started'] = False
                    st.rerun()
                
                if reset_pressed:
                    st.session_state['detection_started'] = False
                    st.session_state['violation_list'] = []
                    st.rerun()

                st_frame = st.empty()
                existing_ids = set(v['ID'] for v in st.session_state['violation_list'])
                captured_ids = existing_ids.copy()
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: 
                        st.warning("Video stream ended.")
                        st.session_state['detection_started'] = False
                        st.rerun()
                        break
                    
                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue 
                    
                    # process_frame now handles smart resize internally
                    processed_frame, found = process_frame(frame, model, points, captured_ids, is_video=True)
                    
                    st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                cap.release()