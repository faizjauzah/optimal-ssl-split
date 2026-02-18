from ultralytics import YOLO

# ================= KONFIGURASI =================
DATA_YAML = r"C:\Users\mfaiz\Documents\dev\tugas_akhir_final\dataset\scenario_L20_U80\data_student.yaml" # File YAML dataset Student (gabungan labeled + pseudo labeled)
PROJECT_NAME = "hasilAkhir"
RUN_NAME = "scenario_L20_student" 
# ===============================================

def main():
    # ---------------------------------------------------------
    # STRATEGI NOISY STUDENT: UKURAN MODEL
    # ---------------------------------------------------------
    # Teori Noisy Student menyarankan Student idealnya >= Teacher.
    # Jika Teacher = yolov12n, Student sebaiknya yolov12n atau yolov12s.
    # Jika VRAM Anda cukup (misal 6GB+), coba naik ke 'yolov12s.pt'
    # agar kapasitas belajarnya lebih besar untuk menyerap noise.
    
    # Kita load model 'yolov12n.pt' (Pretrained COCO) lagi dari awal.
    # Kenapa tidak load 'best.pt' guru? 
    # Agar Murid tidak terjebak pada bias/kebodohan Guru sejak awal (Cold Start).
    model_size = "yolov12n.pt" 
    
    # Load arsitektur model dari pretrained weights
    model = YOLO(model_size) 

    # Informasi konfigurasi training
    print(f"🚀 Memulai Training STUDENT (Noisy Student Iterasi 1)")
    print(f"🧠 Model: {model_size}")
    print(f"📂 Config: {DATA_YAML}")
    print(f"🔥 Mode: Strong Augmentation (Noise Injection)")

    # =========================================================
    # TRAINING MODEL STUDENT
    # =========================================================
    model.train(
        data=DATA_YAML,
        
        # ==============================
        # Epoch & Batch Configuration
        # ==============================
        epochs=200, # Jumlah total iterasi training. Lebih besar = model belajar lebih lama    
        imgsz=640, # Resolusi input gambar. 640 adalah standar keseimbangan akurasi vs kecepatan
        batch=16, # Jumlah gambar per iterasi. Turunkan jika VRAM tidak cukup    
        
        # ==============================
        # Optimizer & Learning Strategy
        # ==============================
        optimizer='auto', # Sistem otomatis memilih optimizer terbaik (biasanya SGD untuk stabilitas)
        lr0=0.01, # Learning rate awal. Terlalu besar → training tidak stabil, terlalu kecil → lambat belajar
        lrf=0.01, # Rasio learning rate akhir (scheduler decay)
        momentum=0.937, #Membantu optimizer mempertahankan arah update agar konvergen lebih cepat
        weight_decay=0.0005, # Regularisasi L2 untuk mencegah bobot terlalu besar (overfitting)
        warmup_epochs=5.0, # Warmup sedikit lebih lama untuk stabilitas awal

        # ==============================
        # NOISE INJECTION (INTI NOISY STUDENT)
        # ==============================
        # Augmentasi kuat memaksa Student belajar pola umum,
        # bukan menghafal pseudo-label yang mungkin salah.
        mosaic=1.0, # Menggabungkan 4 gambar → sangat efektif untuk objek kecil/padat
        close_mosaic=10, # Dimatikan di akhir training agar model adaptasi ke data asli
        mixup=0.1, # Campur 2 gambar → meningkatkan generalisasi fitur
        
        # ==============================
        # Augmentasi Geometris & Warna
        # ==============================
        hsv_h=0.015, # Perubahan Hue
        hsv_s=0.7, # Perubahan Saturasi (penting untuk kemasan produk)
        hsv_v=0.4, # Perubahan Value/Brightness
        degrees=0.0, # Rotasi (0.0 biasanya cukup untuk rak ritel yang tegak)
        translate=0.1, # Geser gambar
        scale=0.5, # Scaling (agar belajar deteksi objek ukuran variatif)
        fliplr=0.5, # Flip horizontal (boleh untuk produk ritel)
        
        # ==============================
        # Sistem, Logging, dan Penyimpanan
        # ==============================
        project=PROJECT_NAME, # Folder utama penyimpanan hasil training
        name=RUN_NAME, # Nama subfolder run eksperimen
        device=0, # GPU index (0 = GPU pertama, 'cpu' jika tanpa GPU).  
        patience=30, # Early stopping → stop jika tidak improve
        exist_ok=True, # Izinkan overwrite folder jika sudah ada
        verbose=True, # Tampilkan log detail selama training
        workers=4, # Thread loader data (stabil di Windows)
        
        # ==============================
        # Scheduler Optimization
        # ==============================
        # cosine_lr=True membantu konvergensi yang lebih halus di akhir
        cos_lr=True # Cosine learning rate decay → penurunan LR lebih halus di akhir training
    )

    print("✅ Training Student Selesai!")
    
    # Validasi Akhir
    # Validasi pada data TEST (Data murni yang tidak pernah dilihat saat training)
    # Pastikan di data_student.yaml, bagian 'val' atau 'test' mengarah ke folder test
    print("📊 Melakukan Validasi pada Test Set...")
    metrics = model.val(split='test')
    print(f"📊 Student mAP@50: {metrics.box.map50}")
    print(f"📊 Student mAP@50-95: {metrics.box.map}")

if __name__ == '__main__':
    main()