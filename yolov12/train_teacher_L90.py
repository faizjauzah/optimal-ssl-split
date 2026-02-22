from ultralytics import YOLO
import torch

# ================= KONFIGURASI =================
DATA_YAML = "dataset/scenario_L90_U10/data.yaml"
PROJECT_NAME = "hasilAkhir"
RUN_NAME = "scenario_L90_teacher" 
# ===============================================

def main():
    # Cek ketersediaan GPU & FlashAttention (Penting untuk YOLOv12)
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Warning: Training on CPU will be very slow for YOLOv12")

    # 1. Load Model
    # YOLOv12 menggunakan arsitektur Area Attention & R-ELAN yang butuh resource
    model = YOLO("yolov12n.pt") 

    print(f"🚀 Memulai Training Baseline (Teacher) untuk: {RUN_NAME}")
    print(f"📂 Config: {DATA_YAML}")
    
    # 2. Mulai Training dengan Hyperparameter YOLOv12
    # Setting default dapat dilihat di yolov12\ultralytics\cfg\default.yaml
    results = model.train(
        data=DATA_YAML, # Path ke file konfigurasi dataset (train/val/test + class names)
        
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
        lr0=0.01,  # Learning rate awal. Terlalu besar → training tidak stabil, terlalu kecil → lambat belajar
        lrf=0.01,  # Rasio learning rate akhir (scheduler decay)
        momentum=0.937, #Membantu optimizer mempertahankan arah update agar konvergen lebih cepat
        weight_decay=0.0005, # Regularisasi L2 untuk mencegah bobot terlalu besar (overfitting)
        warmup_epochs=5.0, # Warmup sedikit lebih lama untuk stabilitas awal
        
        # ==============================
        # Regularization (Penting untuk dataset kecil)
        # ==============================
        dropout=0.1, # Menonaktifkan sebagian neuron saat training → mencegah model hafal data
        
        # ==============================
        # Sistem, Logging, dan Penyimpanan
        # ==============================
        project=PROJECT_NAME, # Folder utama penyimpanan hasil training
        name=RUN_NAME, # Nama subfolder run eksperimen
        device=0, # GPU index (0 = GPU pertama, 'cpu' jika tanpa GPU).
        patience=30, # Early stopping: stop jika tidak ada peningkatan setelah 30 epoch
        exist_ok=True, # Izinkan overwrite folder jika sudah ada
        verbose=True, # Tampilkan log detail selama training
        workers=4, # Jumlah thread loader data
        
        # ==============================
        # Augmentasi Data (Meningkatkan Ketahanan Model Teacher)
        # ==============================
        # Tujuan augmentasi: membuat model belajar dari variasi data buatan sehingga
        # performa tetap bagus saat menghadapi gambar dunia nyata yang berbeda kondisi.
        # Ini sangat penting untuk Teacher model karena hasil prediksinya akan dipakai sebagai pseudo-label untuk data tanpa label.
        mosaic=1.0, # Menggabungkan 4 gambar jadi 1 → memperkaya variasi posisi objek & sangat membantu deteksi objek kecil
        close_mosaic=10, # Menonaktifkan mosaic di 10 epoch terakhir agar model beradaptasi dengan distribusi gambar asli
        mixup=0.0, # Mixup dimatikan karena model Nano kurang stabil dengan augmentasi campuran kompleks
        hsv_h=0.015, # Hue shift → variasi warna ringan agar model tahan perubahan tone warna produk
        hsv_s=0.7, # Saturation → variasi intensitas warna supaya model tidak bergantung pada warna spesifik
        hsv_v=0.4, # Value → variasi pencahayaan agar model robust terhadap kondisi terang/gelap
        
        # ==============================
        # Performa Training
        # ==============================
        # Cache dataset ke RAM agar training cepat (jika RAM cukup)
        cache=False # Ubah ke True jika RAM ≥16GB → mempercepat training, tapi butuh memori besar
    )

    print("✅ Training Selesai!")
    print(f"📊 Hasil tersimpan di: runs/detect/{PROJECT_NAME}/{RUN_NAME}")
    # Validasi model terbaik pada data test
    metrics = model.val(split='test')
    print(f"📊 mAP@50: {metrics.box.map50}")
    print(f"📊 mAP@50-95: {metrics.box.map}")

if __name__ == '__main__':
    main()