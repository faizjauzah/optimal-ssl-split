import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(page_title="Demo YOLOv12 SSL - Ritel", layout="wide")

st.title("Demo Deteksi Objek Ritel: YOLOv12 + Noisy Student")
st.markdown("Geser slider untuk membandingkan hasil deteksi dan grafik pelatihan dari berbagai skenario data berlabel.")

# Slider untuk memilih rasio data berlabel
rasio_label = st.slider("Rasio Data Berlabel (%)", min_value=10, max_value=90, step=10)

# Mengamankan jalur direktori aktif
current_dir = os.path.dirname(os.path.abspath(__file__))

# Fungsi untuk memuat model dengan caching
@st.cache_resource
def load_model(rasio):
    model_path = os.path.join(current_dir, "weights", f"best_{rasio}.pt")
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        return None

# Memuat model
model = load_model(rasio_label)

st.divider()

# Membuat Tab agar UI rapi
tab1, tab2, tab3 = st.tabs([
    "🖼️ Demo Inferensi Gambar", 
    "📊 Grafik Metrik Pelatihan", 
    "📈 Perbandingan Semua Skenario"
])

# ==========================================
# TAB 1: DEMO INFERENSI
# ==========================================
with tab1:
    st.header(f"Pengujian Visual (Model {rasio_label}%)")
    uploaded_file = st.file_uploader("Unggah gambar rak ritel (jpg/png)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if model is None:
            st.error(f"File model untuk rasio {rasio_label}% tidak ditemukan di folder weights!")
        else:
            col1, col2 = st.columns(2)
            image = Image.open(uploaded_file)
            
            with col1:
                st.subheader("Gambar Asli")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Hasil Deteksi")
                if st.button("Jalankan Inferensi", key="btn_infer"):
                    with st.spinner("Model sedang memproses gambar..."):
                        results = model.predict(source=image, conf=0.25)
                        res_plotted = results[0].plot()[:, :, ::-1] 
                        st.image(res_plotted, use_container_width=True)
                        st.success(f"Mendeteksi {len(results[0].boxes)} objek.")

# ==========================================
# TAB 2: METRIK PELATIHAN (INTERAKTIF DARI CSV)
# ==========================================
with tab2:
    st.header(f"Performa Model selama Pelatihan (Skenario {rasio_label}%)")
    
    # Mencari file CSV berdasarkan posisi slider
    csv_path = os.path.join(current_dir, "metrics", f"results_{rasio_label}.csv")
    
    if os.path.exists(csv_path):
        # Membaca data CSV
        df = pd.read_csv(csv_path)
        
        # Membersihkan spasi pada nama kolom bawaan YOLO
        df.columns = df.columns.str.strip()
        
        # Mengatur kolom 'epoch' sebagai sumbu X (index)
        if 'epoch' in df.columns:
            df.set_index('epoch', inplace=True)
        
        # --- BAGIAN 1: METRIK PERFORMA (mAP, Precision, Recall) ---
        st.markdown("### 🎯 Metrik Performa Deteksi")
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.write("**Grafik mAP (Mean Average Precision)**")
            kolom_map = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']
            if all(k in df.columns for k in kolom_map):
                st.line_chart(df[kolom_map])
            else:
                st.warning("Kolom mAP tidak ditemukan.")
                
        with col_perf2:
            st.write("**Grafik Precision & Recall**")
            kolom_pr = ['metrics/precision(B)', 'metrics/recall(B)']
            if all(k in df.columns for k in kolom_pr):
                st.line_chart(df[kolom_pr])
            else:
                st.warning("Kolom Precision/Recall tidak ditemukan.")

        st.divider()

        # --- BAGIAN 2: ANALISIS LOSS (Box, Cls, DFL) ---
        st.markdown("### 📉 Analisis Loss (Train vs Validation)")
        col_loss1, col_loss2, col_loss3 = st.columns(3)
        
        with col_loss1:
            st.write("**Box Loss**")
            kolom_box = ['train/box_loss', 'val/box_loss']
            if all(k in df.columns for k in kolom_box):
                st.line_chart(df[kolom_box])
                
        with col_loss2:
            st.write("**Classification (Cls) Loss**")
            kolom_cls = ['train/cls_loss', 'val/cls_loss']
            if all(k in df.columns for k in kolom_cls):
                st.line_chart(df[kolom_cls])
                
        with col_loss3:
            st.write("**Distribution Focal (DFL) Loss**")
            kolom_dfl = ['train/dfl_loss', 'val/dfl_loss']
            if all(k in df.columns for k in kolom_dfl):
                st.line_chart(df[kolom_dfl])
        
        st.divider()
        
        # --- INSIGHT UNTUK BAB 4 ---
        st.info(f"""
        💡 **Catatan Analisis Mentor untuk Bab 4 (Skenario {rasio_label}%):**
        * **Precision vs Recall**: Pada dataset ritel, apakah model lebih sering menebak salah (*False Positive*, Precision turun) atau lebih sering melewatkan objek (*False Negative*, Recall turun)?
        * **Cls Loss**: Mengukur seberapa baik model membedakan kelas objek (misal: botol air vs kaleng soda). Apakah penambahan *pseudo-label* membuat model kebingungan membedakan kelas?
        * **DFL Loss & Box Loss**: Mengukur ketepatan letak *bounding box*. Jika *val loss* mulai naik di akhir *epoch*, periksa kembali apakah label sisa dari metode SSL kurang presisi posisinya.
        """)
            
    else:
        st.error(f"File CSV tidak ditemukan di jalur: {csv_path}. Pastikan file sudah dicopy dan dinamai dengan benar.")

# ==========================================
# TAB 3: PERBANDINGAN SEMUA SKENARIO (SWEET SPOT ANALYSIS)
# ==========================================
with tab3:
    st.header("Analisis Komparatif Performa Model (SSL YOLOv12)")
    st.markdown("""
    Bagian ini membandingkan metrik mAP dari seluruh skenario rasio data berlabel (10% - 90%) 
    untuk melihat pengaruh metode *Noisy Student* secara keseluruhan.
    """)

    # Daftar semua rasio yang ingin dibandingkan
    semua_rasio = list(range(10, 100, 10))
    
    # List untuk menampung data gabungan
    data_map50 = {}
    data_map50_95 = {}

    with st.spinner("Mengompilasi data dari seluruh skenario..."):
        for rasio in semua_rasio:
            csv_path_all = os.path.join(current_dir, "metrics", f"results_{rasio}.csv")
            
            if os.path.exists(csv_path_all):
                df_temp = pd.read_csv(csv_path_all)
                df_temp.columns = df_temp.columns.str.strip()
                
                # Gunakan epoch sebagai index agar sinkron di grafik
                if 'epoch' in df_temp.columns:
                    # Ambil kolom mAP50 dan mAP50-95, simpan dengan nama kolom sesuai rasio
                    data_map50[f"Rasio {rasio}%"] = df_temp['metrics/mAP50(B)']
                    data_map50_95[f"Rasio {rasio}%"] = df_temp['metrics/mAP50-95(B)']

    if data_map50:
        # Konversi dictionary ke DataFrame
        df_comparison_50 = pd.DataFrame(data_map50)
        df_comparison_95 = pd.DataFrame(data_map50_95)
        
        col_comp1, col_comp2 = st.columns(2)

        with col_comp1:
            st.subheader("Grafik Perbandingan mAP50")
            st.line_chart(df_comparison_50)
            st.caption("Tren kenaikan mAP50 untuk setiap rasio data berlabel.")

        with col_comp2:
            st.subheader("Grafik Perbandingan mAP50-95")
            st.line_chart(df_comparison_95)
            st.caption("Tren stabilitas deteksi pada berbagai tingkat IoU (mAP50-95).")

        st.divider()
        
        # Ringkasan Akhir untuk Bab 4/5
        st.success("### 🎓 Insight Mentor untuk Analisis 'Sweet Spot'")
        
        # Mencari nilai max dari masing-masing skenario untuk tabel ringkasan
        summary_data = []
        for col in df_comparison_50.columns:
            summary_data.append({
                "Skenario": col,
                "Max mAP50": df_comparison_50[col].max(),
                "Max mAP50-95": df_comparison_95[col].max()
            })
        
        st.table(pd.DataFrame(summary_data))
        
        st.info("""
        **Tips Menulis Bab 4:**
        1. **Analisis Efisiensi**: Perhatikan selisih antara Rasio 30% dan Rasio 90%. Jika perbedaannya sangat tipis (misal < 2%), maka Anda bisa berargumen bahwa metode SSL Noisy Student sangat efektif menghemat biaya pelabelan.
        2. **Fenomena Plateau**: Apakah ada rasio tertentu di mana performa tidak lagi naik secara signifikan? Itulah *sweet spot* Anda.
        3. **Kestabilan**: Jika grafik mAP50-95 sangat fluktuatif di rasio rendah (10-20%), artinya *pseudo-label* masih mengandung banyak 'noise' yang mengganggu lokalisasi objek.
        """)
    else:
        st.warning("Data CSV untuk perbandingan tidak ditemukan. Pastikan file 'results_10.csv' hingga 'results_90.csv' tersedia di folder metrics.")