import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="YOLO Vehicle Classifier", page_icon="🚗", layout="centered")
st.title("🚗 Klasifikasi Kendaraan Real-Time")
st.write("Aplikasi ini berjalan di Google Chrome. Unggah video MP4 untuk mendeteksi kendaraan.")

@st.cache_resource
def load_model():
    # Menggunakan yolov8n (Nano) karena ramah batas RAM hosting gratis
    return YOLO("yolov8n.pt")

model = load_model()
uploaded_file = st.file_uploader("Pilih file video (.mp4)", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    video = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    st.info("Sedang memproses video...")
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        # Prediksi kendaraan (sepeda, mobil, motor, bus, truk)
        results = model.predict(frame, classes=[1, 2, 3, 5, 7], conf=0.25, verbose=False)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
        
    video.release()
    st.success("Selesai diproses!")