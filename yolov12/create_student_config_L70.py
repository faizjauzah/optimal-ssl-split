import yaml
import os
import glob

# ================= KONFIGURASI =================
BASE_PROJECT_DIR = r"C:\Users\mfaiz\Documents\dev\tugas_akhir_final\dataset"
SCENARIO_NAME = "scenario_L70_U30"  # Bisa diganti sesuai skenario
# ===============================================

def check_labels_exist(image_path):
    """
    Memastikan folder labels ada dan tidak kosong.
    Asumsi struktur YOLO standar: 
    dataset/images/.. -> dataset/labels/..
    """
    # Mengubah path 'images' menjadi 'labels'
    if "images" not in image_path:
        print(f"⚠️ Warning: Path '{image_path}' tidak mengandung kata 'images'. Cek struktur folder.")
        return False
        
    label_path = image_path.replace("images", "labels")
    
    if not os.path.exists(label_path):
        print(f"❌ Error: Folder label tidak ditemukan di: {label_path}")
        return False
    
    # Hitung jumlah file .txt
    txt_files = glob.glob(os.path.join(label_path, "*.txt"))
    count = len(txt_files)
    
    if count == 0:
        print(f"⚠️ Warning: Folder label ditemukan tapi KOSONG di: {label_path}")
        print("   Apakah Anda sudah menjalankan proses Pseudo-Labeling dengan Teacher?")
        return False
        
    # Mengambil nama folder parent + nama folder ujung (contoh: 'train_labeled/labels')
    parent_dir = os.path.basename(os.path.dirname(label_path))
    current_dir = os.path.basename(label_path)
    folder_name = f"{parent_dir}/{current_dir}"
    
    print(f"✅ Validasi OK: Ditemukan {count} label di '{folder_name}'")
    
    return True

def create_student_yaml():
    scenario_dir = os.path.join(BASE_PROJECT_DIR, SCENARIO_NAME)
    print(f"Mempersiapkan konfigurasi Student untuk: {SCENARIO_NAME}...")
    
    # 1. Baca data.yaml asli
    source_yaml = os.path.join(scenario_dir, "data.yaml")
    if not os.path.exists(source_yaml):
        print(f"❌ Error: data.yaml tidak ditemukan di {scenario_dir}")
        return

    with open(source_yaml, 'r') as f:
        data = yaml.safe_load(f)

    # 2. Tentukan Path
    path_labeled = os.path.join(scenario_dir, "train_labeled", "images")
    path_unlabeled = os.path.join(scenario_dir, "train_unlabeled", "images")

    # 3. Validasi Label (Sangat Penting untuk SSL)
    print("--- Memeriksa Integritas Data ---")
    labeled_ok = check_labels_exist(path_labeled)
    unlabeled_ok = check_labels_exist(path_unlabeled)

    if not unlabeled_ok:
        print("\n⛔ STOP: Data unlabeled belum memiliki pseudo-labels atau folder salah.")
        print("   Jangan lanjutkan training Student karena akan dianggap background images.")
        return

    # 4. Modifikasi Config
    # Menggabungkan Labeled + Unlabeled (Pseudo-labeled)
    data['train'] = [path_labeled, path_unlabeled]
    
    # Pastikan val/test mengarah ke set validasi yang benar
    data['val'] = os.path.join(scenario_dir, "val", "images") 
    data['test'] = os.path.join(scenario_dir, "test", "images")

    # 5. Simpan
    output_yaml = os.path.join(scenario_dir, "data_student.yaml")
    with open(output_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
        
    print(f"\n✅ Sukses! File konfigurasi tersimpan: {output_yaml}")
    print(f"   Mode: Semi-Supervised (Labeled + Pseudo-labeled)")

if __name__ == "__main__":
    create_student_yaml()