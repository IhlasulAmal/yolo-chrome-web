import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time
import pandas as pd

# Konfigurasi Halaman Utama
st.set_page_config(page_title="YOLO Vehicle Classifier", page_icon="🚗", layout="wide")

st.title("🚗 Sistem Klasifikasi Kendaraan Berbasis Web")
st.write("Aplikasi deteksi kendaraan menggunakan model YOLOv8. Pilih metode input di bilah samping (sidebar).")

# 1. Load Model YOLO (Cached agar efisien)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Pemetaan ID Kelas COCO ke Nama Bahasa Indonesia untuk Log & Counter
CLASS_NAMES = {1: "Sepeda", 2: "Mobil", 3: "Motor", 5: "Bus", 7: "Truk"}
VEHICLE_CLASSES = list(CLASS_NAMES.keys())

# 2. Menu Interaktif di Sidebar
st.sidebar.header("🕹️ Menu Kontrol")
menu_pilihan = st.sidebar.radio(
    "Pilih Metode Input:",
    ("Unggah Video (.mp4)", "Kamera Langsung (Live Camera)")
)

# Inisialisasi Tempat Tampilan Visual & Data (Menggunakan Layout Kolom)
kolom_kiri, kolom_kanan = st.columns([2, 1])

with kolom_kiri:
    st.subheader("📺 Visualisasi Deteksi")
    st_frame = st.empty()  # Tempat untuk merender video/kamera

with kolom_kanan:
    st.subheader("📊 Statistik Kendaraan")
    st_counter = st.empty()  # Tempat menampilkan jumlah total kendaraan saat ini
    st.subheader("📜 Log Aktivitas (Terbaru)")
    st_log = st.empty()  # Tempat menampilkan log kendaraan yang lewat

# List global untuk menampung riwayat log selama aplikasi berjalan (sementara)
if "log_history" not in st.session_state:
    st.session_state.log_history = []

# -------------------------------------------------------------------------
# OPSI A: UNGGAH VIDEO (VERSI SUPER SMOOTH & ANTI-FLICKER)
# -------------------------------------------------------------------------
if menu_pilihan == "Unggah Video (.mp4)":
    uploaded_file = st.file_uploader("Pilih file video (.mp4)", type=["mp4"])

    if uploaded_file is not None:
        # Simpan file sementara
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # Ambil informasi total frame video untuk timeline
        cap_info = cv2.VideoCapture(tfile.name)
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_info.release()
        
        # Kontrol Interaktif di Sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎮 Kontrol Pemutar")
        
        # Menggunakan checkbox sebagai switch Play/Stop agar tidak memicu rerun massal
        play_mode = st.sidebar.checkbox("▶️ Jalankan Video (Play)", value=False)

        # Garis Waktu / Timeline Slider
        timeline = st.slider(
            "🎞️ Jalankan Manual / Geser Waktu (Gunakan saat Video STOP)", 
            min_value=0, 
            max_value=total_frames - 1, 
            value=0
        )
        
        # Mulai Buka Video
        video = cv2.VideoCapture(tfile.name)
        
        # KONDISI 1: JIKA TOMBOL PLAY AKTIF (Video berjalan mulus lewat internal loop)
        if play_mode:
            st.info("Video sedang berjalan... Hilangkan centang 'Jalankan Video' untuk Pause.")
            
            # Set posisi awal video berdasarkan titik terakhir slider
            video.set(cv2.CAP_PROP_POS_FRAMES, timeline)
            
            # Kunci Rahasia: Perulangan internal 'while' TANPA st.rerun() agar tidak berkedip
            while video.isOpened() and play_mode:
                ret, frame = video.read()
                if not ret:
                    break
                
                # Optimasi ukuran frame (wajib agar streaming cloud lancar)
                frame = cv2.resize(frame, (640, 360)) 
                
                # Prediksi YOLO
                results = model.predict(frame, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Tampilkan ke Web (Hanya komponen gambar ini yang diperbarui, komponen lain diam)
                st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
                
                # --- PROSES LOG DAN COUNTER ---
                current_counts = {name: 0 for name in CLASS_NAMES.values()}
                boxes = results[0].boxes
                
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    if class_id in CLASS_NAMES:
                        nama_kendaraan = CLASS_NAMES[class_id]
                        current_counts[nama_kendaraan] += 1
                        
                        timestamp = time.strftime('%H:%M:%S')
                        st.session_state.log_history.insert(0, {"Waktu": timestamp, "Jenis Kendaraan": nama_kendaraan})
                
                # Update Statistik & Log tabel kanan secara berkala
                df_count = pd.DataFrame(list(current_counts.items()), columns=["Jenis", "Jumlah Terdeteksi"])
                st_counter.dataframe(df_count, use_container_width=True, hide_index=True)
                
                if st.session_state.log_history:
                    df_log = pd.DataFrame(st.session_state.log_history[:5])
                    st_log.dataframe(df_log, use_container_width=True, hide_index=True)
                
                # Jeda tipis untuk menyamakan dengan kecepatan FPS asli video
                time.sleep(0.03) 
                
        # KONDISI 2: JIKA VIDEO PAUSE / STOP (Menampilkan gambar statis sesuai geseran slider)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, timeline)
            ret, frame = video.read()
            if ret:
                frame = cv2.resize(frame, (640, 360))
                results = model.predict(frame, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(annotated_frame, channels="RGB", use_container_width=True)

        video.release()
# -------------------------------------------------------------------------
# OPSI B: KAMERA LANGSUNG (LIVE CAMERA)
# -------------------------------------------------------------------------
elif menu_pilihan == "Kamera Langsung (Live Camera)":
    jalankan_kamera = st.sidebar.checkbox("Nyalakan Kamera", value=False)
    
    if jalankan_kamera:
        # Angka 0 merujuk pada webcam bawaan laptop/PC lokal kamu
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Gagal membuka kamera. Pastikan kamera tidak dipakai aplikasi lain.")
            jalankan_kamera = False
        
        while jalankan_kamera:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal mengambil gambar dari kamera.")
                break
                
            # Jalankan Prediksi YOLO pada Live Frame
            results = model.predict(frame, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Update Tampilan Gambar Kamera
            st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
            
            # --- PROSES LOG DAN COUNTER LIVE ---
            current_counts = {name: 0 for name in CLASS_NAMES.values()}
            boxes = results[0].boxes
            
            for box in boxes:
                class_id = int(box.cls[0].item())
                if class_id in CLASS_NAMES:
                    nama_kendaraan = CLASS_NAMES[class_id]
                    current_counts[nama_kendaraan] += 1
                    
                    timestamp = time.strftime('%H:%M:%S')
                    st.session_state.log_history.insert(0, {"Waktu": timestamp, "Jenis Kendaraan": nama_kendaraan})
            
            # Update Dataframe Counter di Kolom Kanan
            df_count = pd.DataFrame(list(current_counts.items()), columns=["Jenis", "Jumlah Terdeteksi"])
            st_counter.dataframe(df_count, use_container_width=True, hide_index=True)
            
            # Update Dataframe Log Terbaru
            if st.session_state.log_history:
                df_log = pd.DataFrame(st.session_state.log_history[:10])
                st_log.dataframe(df_log, use_container_width=True, hide_index=True)
                
            # Berikan jeda sangat kecil agar Streamlit sempat memproses komponen UI lainnya
            time.sleep(0.01)
            
        cap.release()
    else:
        st_frame.info("Kamera mati. Centang opsi 'Nyalakan Kamera' pada bilah kontrol di samping untuk memulai.")
