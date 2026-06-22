try:
    import streamlit as st
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The required package 'streamlit' is not installed. Install it with `pip install streamlit` and rerun the app."
    ) from e

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import io

# Set page configuration with a modern dashboard look
st.set_page_config(
    page_title="Deteksi Perlengkapan Berkendara - YOLOv12",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .title-text {
        font-family: 'Outfit', sans-serif;
        color: #1E293B;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle-text {
        font-family: 'Inter', sans-serif;
        color: #64748B;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .status-aman {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #BCF0DA;
    }
    .status-bahaya {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #FBD5D5;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<h1 class='title-text'>🏍️ Deteksi Perlengkapan Berkendara</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Sistem Keamanan Berkendara Motor Otomatis berbasis Deep Learning YOLOv12</p>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/144/motorcycle.png", width=100)
st.sidebar.title("⚙️ Pengaturan Model")

# Choose Model File
model_path = st.sidebar.text_input("Path Model (.pt)", value="runs/detect/train/weights/best.pt", help="Masukkan path file hasil training (misal: runs/detect/train/weights/best.pt atau best.pt")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.45, step=0.05, help="Batas kepercayaan deteksi")
iou_threshold = st.sidebar.slider("IOU Threshold", min_value=0.1, max_value=1.0, value=0.45, step=0.05, help="NMS Overlap Threshold")

# Load YOLO model
@st.cache_resource
def load_model(path):
    try:
        # If best.pt does not exist, use a pre-trained model for demo
        if not os.path.exists(path):
            st.sidebar.warning(f"File '{path}' tidak ditemukan. Menggunakan model bawaan 'yolo12s.pt' sebagai fallback untuk testing.")
            return YOLO("yolo12s.pt")
        return YOLO(path)
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

model = load_model(model_path)

# Navigation / Mode Selection
app_mode = st.sidebar.selectbox("Pilih Input Media", ["Unggah Gambar", "Unggah Video", "Kamera Real-time (Webcam)"])

if model is not None:
    if app_mode == "Unggah Gambar":
        st.header("📸 Analisis Gambar")
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Predict
            with st.spinner("Menganalisis gambar dengan YOLOv12..."):
                results = model.predict(img_array, conf=confidence_threshold, iou=iou_threshold)
            
            # Class definitions
            # 0: Helmet, 1: Number Plate, 2: Rearview Mirror (depends on your roboflow dataset order)
            detected_classes = []
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                detected_classes.append(cls_name)
            
            # Draw results on image
            annotated_image = results[0].plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Display layout columns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Gambar Asli")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Hasil Deteksi")
                st.image(annotated_image, use_column_width=True)
                
            # Stats & Analysis
            st.subheader("📊 Hasil Analisis Keselamatan")
            
            # Counting instances
            counts = {name: detected_classes.count(name) for name in set(detected_classes)}
            
            # Display counts
            stat_cols = st.columns(3)
            with stat_cols[0]:
                helmet_count = counts.get("Helmet", 0) + counts.get("helmet", 0)
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>🪖 Helm</h3>
                    <h2 style='color:#3B82F6;'>{helmet_count} Terdeteksi</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_cols[1]:
                plate_count = counts.get("Number Plate", 0) + counts.get("number plate", 0) + counts.get("Number-Plate", 0)
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>💳 Plat Nomor</h3>
                    <h2 style='color:#10B981;'>{plate_count} Terdeteksi</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with stat_cols[2]:
                mirror_count = counts.get("Rearview Mirror", 0) + counts.get("rearview mirror", 0) + counts.get("Rearview-Mirror", 0)
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>🪞 Spion</h3>
                    <h2 style='color:#F59E0B;'>{mirror_count} Terdeteksi</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Safety Status
            st.markdown("<br>", unsafe_allow_html=True)
            if helmet_count > 0:
                st.markdown("<div class='status-aman'>✅ PENGENDARA MEMAKAI HELM (Aman / Memenuhi Syarat)</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status-bahaya'>⚠️ PERINGATAN: Pengendara TIDAK MEMAKAI HELM! (Sangat Bahaya / Melanggar Aturan)</div>", unsafe_allow_html=True)
            
            # Download detected image
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📥 Unduh Hasil Deteksi")
            
            # Convert annotated image to PIL Image for download
            annotated_pil = Image.fromarray(annotated_image)
            
            # Create buffer for image
            img_buffer = io.BytesIO()
            annotated_pil.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            # Get original filename
            original_filename = uploaded_file.name
            output_filename = f"detected_{original_filename.split('.')[0]}.png"
            
            # Download button
            st.download_button(
                label="⬇️ Download Gambar Hasil Deteksi",
                data=img_buffer.getvalue(),
                file_name=output_filename,
                mime="image/png",
                use_container_width=True
            )

    elif app_mode == "Unggah Video":
        st.header("🎥 Analisis Video")
        uploaded_video = st.file_uploader("Pilih file video...", type=["mp4", "avi", "mov"])
        
        if uploaded_video is not None:
            # Save upload video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            tfile.close()
            
            # Open Video capture
            cap = cv2.VideoCapture(tfile.name)
            
            # Placeholder for video frame
            frame_placeholder = st.empty()
            
            # Read and process frames
            st.write("Sedang memproses video frame demi frame... Tekan 'Stop' di pojok kanan atas untuk membatalkan.")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run prediction on frame
                results = model.predict(frame, conf=confidence_threshold, iou=iou_threshold)
                
                # Plot bounding boxes on frame
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update Streamlit frame placeholder
                frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
            
            cap.release()
            os.unlink(tfile.name)
            st.success("Pemrosesan video selesai!")

    elif app_mode == "Kamera Real-time (Webcam)":
        st.header("📹 Kamera Real-time")
        st.write("Gunakan kamera laptop Anda untuk mendeteksi perlengkapan berkendara secara langsung.")
        
        # Streamlit standard camera input is picture-based. For real-time video stream,
        # we can provide instruction on how to run a local OpenCV script or use streamlit_webrtc if available.
        # Below is a simple interactive loop using local OpenCV webcam (works when running locally)
        run_webcam = st.checkbox("Mulai Kamera (Lokal)")
        
        if run_webcam:
            cap = cv2.VideoCapture(0) # Open default camera
            if not cap.isOpened():
                st.error("Gagal membuka kamera. Pastikan webcam terhubung dan tidak sedang digunakan oleh aplikasi lain.")
            else:
                frame_placeholder = st.empty()
                stop_button = st.button("Stop Kamera")
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Gagal membaca frame kamera.")
                        break
                    
                    # Flip frame to act like a mirror
                    frame = cv2.flip(frame, 1)
                    
                    # Predict
                    results = model.predict(frame, conf=confidence_threshold, iou=iou_threshold)
                    annotated_frame = results[0].plot()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Update view
                    frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
                    
                cap.release()
                st.info("Kamera dinonaktifkan.")
else:
    st.error("Inisialisasi model gagal. Silakan periksa file model Anda.")

# Footer info
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94A3B8;'>Sistem Deteksi Perlengkapan Berkendara Motor YOLOv12 - Dibuat untuk Keamanan Jalan Raya</p>", unsafe_allow_html=True)
