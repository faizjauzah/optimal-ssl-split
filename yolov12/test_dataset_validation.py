import os
from glob import glob
import yaml

# Path folder skenario yang baru dibuat
OUTPUT_ROOT = 'dataset' 
# Kita cukup cek satu skenario saja karena folder 'test'-nya sama semua
CHECK_PATH = os.path.join(OUTPUT_ROOT, 'scenario_L10_U90') 

def verify_test_completeness():
    print("🕵️ Memulai Verifikasi Kelengkapan Kelas di Test Set...")
    
    # 1. Baca daftar kelas yang SEHARUSNYA ada (dari data.yaml)
    yaml_path = os.path.join(CHECK_PATH, 'data.yaml')
    if not os.path.exists(yaml_path):
        print("❌ Error: File data.yaml tidak ditemukan. Jalankan generator dulu.")
        return
        
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
        
    # Ambil ID kelas dari YAML
    # Format yaml yolo bisa berupa list nama ['a','b'] atau dict {0:'a', 1:'b'}
    names = data_config.get('names')
    if isinstance(names, dict):
        required_classes = set(names.keys())
    else: # list
        required_classes = set(range(len(names)))
        
    print(f"📋 Total Kelas Wajib: {len(required_classes)} (ID: {sorted(list(required_classes))})")
    
    # 2. Baca apa yang BENAR-BENAR ada di folder Test
    test_labels_dir = os.path.join(CHECK_PATH, 'test', 'labels')
    found_classes = set()
    
    label_files = glob(os.path.join(test_labels_dir, '*.txt'))
    
    for lf in label_files:
        with open(lf, 'r') as f:
            for line in f:
                parts = line.split()
                if parts:
                    found_classes.add(int(parts[0]))
    
    print(f"✅ Kelas Ditemukan di Test Set: {len(found_classes)} (ID: {sorted(list(found_classes))})")
    
    # 3. Bandingkan
    missing = required_classes - found_classes
    
    if len(missing) == 0:
        print("\n🎉 HASIL: VALID! Test Set memiliki representasi SEMUA kelas.")
        print("Anda aman untuk lanjut ke tahap Training.")
    else:
        print(f"\n⚠️ HASIL: TIDAK LENGKAP! Kelas yang hilang: {missing}")
        print("Saran: Dataset perlu di-generate ulang dengan 'Smart Splitting'.")

verify_test_completeness()