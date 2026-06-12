import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Konfigurasi Halaman
st.set_page_config(page_title="YOLO Vehicle Classifier Pro", page_icon="🚗", layout="wide")
st.title("🚗 Sistem Klasifikasi Kendaraan Pro (Google Chrome Version)")

# -------------------------------------------------------------------------
# 1. SIDEBAR: PEMILIHAN MODEL & INPUT
# -------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Sistem")

# Fitur ganti-ganti model secara dinamis
pilihan_model = st.sidebar.selectbox(
    "Pilih Model YOLOv8:",
    ("YOLOv8 Nano (Super Cepat)", "YOLOv8 Small (Seimbang)", "YOLOv8 Medium (Lebih Akurat)")
)

# Pemetaan file model
MODEL_MAP = {
    "YOLOv8 Nano (Super Cepat)": "yolov8n.pt",
    "YOLOv8 Small (Seimbang)": "yolov8s.pt",
    "YOLOv8 Medium (Lebih Akurat)": "yolov8m.pt"
}

@st.cache_resource
def load_model(model_name):
    return YOLO(MODEL_MAP[model_name])

model = load_model(pilihan_model)
st.sidebar.success(# Menggunakan model: {MODEL_MAP[pilihan_model]}`)

# Menu Metode Input
menu_pilihan = st.sidebar.radio(
    "Pilih Metode Input:",
    ("Unggah Video (.mp4)", "Kamera Langsung (Live Camera)")
)

# Konfigurasi Kelas Kendaraan
CLASS_NAMES = {1: "Sepeda", 2: "Mobil", 3: "Motor", 5: "Bus", 7: "Truk"}
VEHICLE_CLASSES = list(CLASS_NAMES.keys())

# Layout Utama: 2 Kolom (Visualisasi Kiri, Statistik Kanan)
kolom_kiri, kolom_kanan = st.columns([2, 1])

# -------------------------------------------------------------------------
# OPSI A: UNGGAH VIDEO (MENGGUNAKAN HTML5 NATIVE PLAYER - 100% SMOOTH)
# -------------------------------------------------------------------------
if menu_pilihan == "Unggah Video (.mp4)":
    with kolom_kiri:
        st.subheader("📺 Visualisasi Deteksi Video")
        uploaded_file = st.file_uploader("Upload video lalu lintas kamu di sini:", type=["mp4"])
    
    if uploaded_file is not None:
        # Buat file temporer untuk video input dan output
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        
        output_path = os.path.join(tempfile.gettempdir(), "output_yolo.mp4")
        
        # Proses konversi video dengan YOLO di background (hanya sekali proses)
        if st.button("🚀 Mulai Proses Analisis Video"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Setup Video Writer untuk menyimpan hasil anotasi YOLO
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (640, 360))
            
            frame_idx = 0
            log_data = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (640, 360))
                results = model.predict(frame, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()
                
                # Tulis frame ke file output
                out.write(annotated_frame)
                
                # Catat Log Kendaraan
                boxes = results[0].boxes
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    if class_id in CLASS_NAMES:
                        log_data.append({
                            "Waktu Deteksi": f"Frame ke-{frame_idx}",
                            "Jenis Kendaraan": CLASS_NAMES[class_id]
                        })
                
                frame_idx += 1
                progress_bar.progress(int((frame_idx / total_frames) * 100))
                status_text.text(f"Memproses Frame: {frame_idx} / {total_frames}")
            
            cap.release()
            out.release()
            
            # Konversi codec video ke H264 agar bisa diputar langsung di Google Chrome
            status_text.text("Mengoptimalkan video untuk Google Chrome...")
            os.system(f"ffmpeg -y -i {output_path} -vcodec libx264 {output_path}_chrome.mp4")
            
            status_text.success("Analisis Selesai!")
            
            # TAMPILKAN VIDEO DI BAGIAN BAWAH DENGAN TOMBOL & SLIDER BAWAAN CHROME
            with kolom_kiri:
                st.markdown("### 🎞️ Pemutar Video Hasil Deteksi")
                video_file = open(f"{output_path}_chrome.mp4", 'rb')
                video_bytes = video_file.read()
                # st.video otomatis menaruh tombol play, stop, full screen, dan timeline slider di BAWAH video secara native!
                st.video(video_bytes)
            
            # Tampilkan statistik akhir di kolom kanan
            with kolom_kanan:
                st.subheader("📊 Statistik Akhir")
                import pandas as pd
                if log_data:
                    df_log = pd.DataFrame(log_data)
                    st.dataframe(df_log, use_container_width=True, hide_index=True)
                    
                    df_count = df_log["Jenis Kendaraan"].value_counts().reset_index()
                    df_count.columns = ["Jenis Kendaraan", "Total Terhitung"]
                    st.dataframe(df_count, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------------
# OPSI B: KAMERA LANGSUNG (LIVE CAMERA MENGGUNAKAN WEBRTC - 100% WORK DI CHROME)
# -------------------------------------------------------------------------
elif menu_pilihan == "Kamera Langsung (Live Camera)":
    with kolom_kiri:
        st.subheader("📹 Tampilan Live Kamera Browser")
        st.info("Izinkan browser Google Chrome mengakses webcam kamu setelah menekan tombol START di bawah.")
        
        # Tempat penampung data log agar bisa di-update dari dalam callback
        if "live_counts" not in st.session_state:
            st.session_state.live_counts = {"Sepeda": 0, "Mobil": 0, "Motor": 0, "Bus": 0, "Truk": 0}

        # Fungsi pemroses frame kamera WebRTC secara realtime
        def transform(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            # Jalankan deteksi YOLO
            results = model.predict(img, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()
            
            # Hitung objek untuk counter live
            boxes = results[0].boxes
            temp_counts = {"Sepeda": 0, "Mobil": 0, "Motor": 0, "Bus": 0, "Truk": 0}
            for box in boxes:
                class_id = int(box.cls[0].item())
                if class_id in CLASS_NAMES:
                    temp_counts[CLASS_NAMES[class_id]] += 1
            
            st.session_state.live_counts = temp_counts
            
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        # Komponen khusus pemanggil kamera yang support Google Chrome & Web Hosting gratisan
        webrtc_streamer(
            key="yolo-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_frame_callback=transform,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # Tampilkan Counter Live di Kolom Kanan
    with kolom_kanan:
        st.subheader("📊 Kendaraan di Kamera Saat Ini")
        import pandas as pd
        df_live = pd.DataFrame(list(st.session_state.live_counts.items()), columns=["Jenis", "Jumlah Terdeteksi"])
        st.dataframe(df_live, use_container_width=True, hide_index=True)
