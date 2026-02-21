from ultralytics import YOLO
import os
import glob
import shutil
from tqdm import tqdm

# ================= KONFIGURASI =================
# 1. Path Model Teacher (Hasil Training Sebelumnya)
# Pastikan path ini benar mengarah ke best.pt yang baru saja jadi
MODEL_PATH = r"C:\Users\mfaiz\Documents\dev\tugas_akhir_final\hasilAkhir\scenario_L10_teacher\weights\best.pt"

# 2. Path Data Unlabeled (Yang mau ditebak)
# Arahkan ke folder images di dalam train_unlabeled
UNLABELED_IMAGES_DIR = r"C:\Users\mfaiz\Documents\dev\tugas_akhir_final\dataset\scenario_L10_U90\train_unlabeled\images"

# 3. Path Output Label (Akan dibuat otomatis)
# Kita taruh di folder 'labels' pasangannya train_unlabeled
UNLABELED_LABELS_DIR = r"C:\Users\mfaiz\Documents\dev\tugas_akhir_final\dataset\scenario_L10_U90\train_unlabeled\labels"

# --- HYPERPARAMETER SSL (PENTING) ---
# Threshold tinggi = hanya prediksi yakin yang dipakai untuk melatih Student
CONF_THRESHOLD = 0.70 # Confidence minimal prediksi dianggap valid
IOU_THRESHOLD = 0.60 # IoU untuk NMS (menghapus box tumpang tindih)
# ===============================================

def generate_pseudo_labels():
    # Cek apakah model ada
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model tidak ditemukan di {MODEL_PATH}")
        print("Pastikan training baseline sudah selesai!")
        return

    print(f"🚀 Memuat model Guru (Teacher): {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        return

    # Membersihkan folder lama (Mencegah Ghost Labels)
    if os.path.exists(UNLABELED_LABELS_DIR):
        print(f"🧹 Membersihkan folder label lama...")
        shutil.rmtree(UNLABELED_LABELS_DIR)
        
    # Buat ulang folder labels yang 100% kosong
    os.makedirs(UNLABELED_LABELS_DIR, exist_ok=True)
    
    # Cari semua gambar (support jpg, jpeg, png, bmp)
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    for ext in extensions:
        # Recursive globbing bisa lambat, kita pakai join sederhana jika struktur flat
        image_files.extend(glob.glob(os.path.join(UNLABELED_IMAGES_DIR, ext)))
    
    if not image_files:
        print(f"❌ Error: Tidak ada gambar di {UNLABELED_IMAGES_DIR}")
        return

    print(f"📂 Ditemukan {len(image_files)} gambar Unlabeled.")
    print(f"🎯 Target Threshold: Conf={CONF_THRESHOLD}, IoU={IOU_THRESHOLD}")
    print("⏳ Mulai Pseudo-labeling dengan TTA (Test Time Augmentation)...")
    
    count_labels = 0 # total bounding box terdeteksi
    count_empty = 0 # jumlah gambar tanpa objek
    
    # =======================================================
    # Inference Batch Generator
    # =======================================================
    # stream=True → hasil diproses satu per satu (hemat RAM)
    # augment=True → TTA (Test Time Augmentation)
    # TTA membuat prediksi lebih stabil karena model melihat variasi versi gambar
    results = model.predict(
        source=image_files,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        augment=True,  
        save=False,
        stream=True,
        verbose=False
    )

    # =======================================================
    # Loop semua hasil prediksi
    # =======================================================
    for result in tqdm(results, total=len(image_files), desc="Generating Labels"):
        
        img_path = result.path
        file_name = os.path.basename(img_path)
        
        # Nama file label harus sama dengan nama gambar
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(UNLABELED_LABELS_DIR, txt_name)
        
        # Format YOLO: class x_center y_center width height (normalized 0-1)
        labels_str = ""
        
        if len(result.boxes) > 0:
            for box in result.boxes:
                # Ambil class ID
                cls_id = int(box.cls[0].item())
                
                # Ambil koordinat xywh normalisasi
                # xywhn sudah ternormalisasi oleh ultralytics
                x, y, w, h = box.xywhn[0].tolist()
                
                # Safety Clamp: Pastikan tidak ada angka > 1.0 atau < 0.0
                # Kadang augmentasi bikin bounding box sedikit keluar frame
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                # Tambahkan baris label
                labels_str += f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                count_labels += 1
        else:
            # Tidak ada objek terdeteksi
            # Tetap dihitung supaya statistik jelas
            count_empty += 1
            
        # Tulis ke file (File kosong tetap ditulis untuk menandakan "tidak ada objek")
        with open(txt_path, "w") as f:
            f.write(labels_str)

    # =======================================================
    # Output hasil pseudo-labeling
    # =======================================================
    print("\n✅ Pseudo-labeling Selesai!")
    print(f"📄 Label tersimpan di: {UNLABELED_LABELS_DIR}")
    print(f"📊 Statistik:")
    print(f"   - Total Gambar: {len(image_files)}")
    print(f"   - Total Objek Terdeteksi: {count_labels}")
    print(f"   - Gambar Kosong (No Object): {count_empty}")
    print("💡 Langkah selanjutnya: Training Model Student menggunakan data Labeled + Unlabeled (Pseudo).")

if __name__ == "__main__":
    generate_pseudo_labels()