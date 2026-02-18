import os
import shutil
import random
import yaml
import pandas as pd
from glob import glob
from collections import Counter, defaultdict

# ================= KONFIGURASI =================
SOURCE_ROOT = 'datasetGramasi/all_img'
OUTPUT_ROOT = 'dataset'
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TEST_SIZE = 0.1  # 10%
VAL_SIZE = 0.1   # 10%
# ===============================================

# DAFTAR KELAS ASLI
CLASS_NAMES_MAP = {
    0: "DANCOW 1+ Madu Advn ExcNutr 12x1000g N1 ID_REDESIGN3",
    1: "DANCOW 1+ Madu Advn ExcNutr 12x800g ID_REDESIGN3",
    2: "DANCOW 1+ Madu Advn ExcNutr 24x400g N1 ID_REDESIGN3",
    3: "DANCOW 1+ Madu Advn ExcNutr 40x200g ID_REDESIGN3",
    4: "DANCOW 1+ Van Advn ExcNutr 12x1000g ID_REDESIGN3",
    5: "DANCOW 1+ Van Advn ExcNutr 12x800g N1 ID_REDESIGN3",
    6: "DANCOW 1+ Van Advn ExcNutr 24x400g ID_REDESIGN3",
    7: "DANCOW 1+ Van Advn ExcNutr 40x200g N1 ID_REDESIGN3",
    8: "DANCOW 3+ Madu Advn ExcNutr 12x1000g ID_REDESIGN3",
    9: "DANCOW 3+ Madu Advn ExcNutr 12x800g ID_REDESIGN3",
    10: "DANCOW 3+ Madu Advn ExcNutr 24x400g ID_REDESIGN3",
    11: "DANCOW 3+ Van Advn ExcNutr 12x1000g ID_REDESIGN3",
    12: "DANCOW 3+ Van Advn ExcNutr 12x800g ID_REDESIGN3",
    13: "DANCOW 3+ Van Advn ExcNutr 24x400g ID_REDESIGN3",
    14: "DANCOW 5+ Madu Advn ExcNutr 12x1000g ID_REDESIGN3",
    15: "DANCOW 5+ Madu Advn ExcNutr 12x800g ID_REDESIGN3",
    16: "DANCOW 5+ Madu Advn ExcNutr 24x400g ID_REDESIGN3",
    17: "LACTOGEN 1 Happynutri 12x1kg ID_REDESIGN2",
    18: "LACTOGEN 1 Happynutri 12x750g N1 ID_REDESIGN2",
    19: "LACTOGEN 1 Happynutri 24x350g N1 ID_REDESIGN2",
    20: "LACTOGEN 1 Happynutri 40x180g N1 ID_REDESIGN2",
    21: "LACTOGEN 2 Happynutri 12x1kg ID_REDESIGN2",
    22: "LACTOGEN 2 Happynutri 12x750g N1 ID_REDESIGN2",
    23: "LACTOGEN 2 Happynutri 24x350g N1 ID_REDESIGN2",
    24: "LACTOGEN 2 Happynutri 40x180g N1 ID_REDESIGN2",
    25: "LACTOGEN PRO 0-6 mo 12x1kg ID",
    26: "LACTOGEN PRO 0-6 mo 12x735g ID",
    27: "LACTOGEN PRO 0-6 mo 24x350g ID",
    28: "LACTOGEN PRO 0-6 mo 40x180g ID",
    29: "LACTOGEN PRO 6-12 mo 12x1kg ID",
    30: "LACTOGEN PRO 6-12 mo 12x735g ID",
    31: "LACTOGEN PRO 6-12 mo 24x350g ID",
    32: "LACTOGEN PRO 6-12 mo 40x180g ID",
    33: "LACTOGROW 3 Happynutri Honey 12x1kg N1 ID_REDESIGN2",
    34: "LACTOGROW 3 Happynutri Honey 12x750g N1 ID_REDESIGN2",
    35: "LACTOGROW 3 Happynutri Honey 24x350g N1 ID_REDESIGN2",
    36: "LACTOGROW 3 Happynutri Honey 40x180g N1 ID_REDESIGN2",
    37: "LACTOGROW 3 Happynutri Van 12x1kg ID_REDESIGN2",
    38: "LACTOGROW 3 Happynutri Van 12x750g N1 ID_REDESIGN2",
    39: "LACTOGROW 3 Happynutri Van 24x350g N1 ID_REDESIGN2",
    40: "LACTOGROW PRO 1+ Honey 12x1kg ID",
    41: "LACTOGROW PRO 1+ Honey 12x735g ID",
    42: "LACTOGROW PRO 1+ Honey 24x350g ID",
    43: "LACTOGROW PRO 1+ Honey 40x145g ID",
    44: "LACTOGROW PRO 1+ Van 12x1kg ID",
    45: "LACTOGROW PRO 1+ Van 12x735g ID",
    46: "LACTOGROW PRO 1+ Van 24x350g ID"
}

def get_image_classes(label_path):
    classes = set()
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    return classes

def analyze_dataset_content(dataset_list):
    total_bboxes = 0
    class_counts = Counter()
    for _, lbl_path, _ in dataset_list:
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                total_bboxes += len(lines)
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_counts[int(parts[0])] += 1
    return total_bboxes, class_counts

def save_statistics_detailed(path, scenario_name, datasets):
    stats = {}
    total_imgs_global = 0
    total_bboxes_global = 0
    subsets = ['train_labeled', 'train_unlabeled', 'val', 'test']
    
    for subset in subsets:
        data_list = datasets[subset]
        n_imgs = len(data_list)
        n_bboxes, c_counts = analyze_dataset_content(data_list)
        stats[subset] = {'imgs': n_imgs, 'bboxes': n_bboxes, 'class_counts': c_counts}
        total_imgs_global += n_imgs
        total_bboxes_global += n_bboxes

    txt_path = os.path.join(path, 'stats_summary.txt')
    with open(txt_path, 'w') as f:
        # HEADER
        f.write(f"=== DATASET STATISTICS: {scenario_name} ===\n\n")
        
        # 1. IMAGE STATISTICS
        f.write("IMAGE STATISTICS:\n")
        f.write(f"  Total Images (Global):  {total_imgs_global}\n")
        for sub in subsets:
            count = stats[sub]['imgs']
            pct = (count / total_imgs_global * 100) if total_imgs_global else 0
            f.write(f"  {sub:<18}:  {count} ({pct:.2f}%)\n")
        f.write("\n")

        # 2. BOUNDING BOX STATISTICS (GLOBAL) -- [DITAMBAHKAN KEMBALI]
        f.write("BOUNDING BOX STATISTICS (GLOBAL):\n")
        f.write(f"  Total BBoxes (Global):  {total_bboxes_global}\n")
        for sub in subsets:
            count = stats[sub]['bboxes']
            pct = (count / total_bboxes_global * 100) if total_bboxes_global else 0
            f.write(f"  {sub:<18}:  {count} ({pct:.2f}%)\n")
        f.write("\n")

        # 3. BOUNDING BOX PER CLASS TABLE
        f.write("BOUNDING BOX PER CLASS (Total Classes: 47):\n")
        sep_line = "-" * 135
        f.write(sep_line + "\n")
        header = f"{'ID':<4} | {'Class Name':<50} | {'Lab':<6} | {'Unlab':<6} | {'Val':<6} | {'Test':<6} | {'Total':<6} | {'Test %':<8}"
        f.write(header + "\n")
        f.write(sep_line + "\n")
        for cls_id in sorted(CLASS_NAMES_MAP.keys()):
            name = CLASS_NAMES_MAP[cls_id]
            name_disp = (name[:47] + '..') if len(name) > 47 else name
            c_lab = stats['train_labeled']['class_counts'].get(cls_id, 0)
            c_unlab = stats['train_unlabeled']['class_counts'].get(cls_id, 0)
            c_val = stats['val']['class_counts'].get(cls_id, 0)
            c_test = stats['test']['class_counts'].get(cls_id, 0)
            c_total = c_lab + c_unlab + c_val + c_test
            test_pct = (c_test / c_total * 100) if c_total > 0 else 0.0
            row = f"{cls_id:<4} | {name_disp:<50} | {c_lab:<6} | {c_unlab:<6} | {c_val:<6} | {c_test:<6} | {c_total:<6} | {test_pct:>6.2f}%"
            f.write(row + "\n")
        f.write(sep_line + "\n")

    csv_rows = []
    for sub in subsets:
        c_counts = stats[sub]['class_counts']
        for cls_id in sorted(CLASS_NAMES_MAP.keys()):
            csv_rows.append({
                'Scenario': scenario_name,
                'Subset': sub,
                'Class_ID': cls_id,
                'Class_Name': CLASS_NAMES_MAP[cls_id],
                'Count': c_counts.get(cls_id, 0)
            })
    pd.DataFrame(csv_rows).to_csv(os.path.join(path, 'class_distribution.csv'), index=False)

def greedy_stratified_select(data_pool, class_to_images, target_count, all_classes):
    """
    ALGORITMA GREEDY (RAKUS):
    Memilih gambar yang paling 'efisien' (memuat paling banyak kelas baru)
    untuk memastikan semua kelas terwakili dalam jumlah slot gambar yang terbatas.
    """
    selected_set = set()
    covered_classes = set()
    pool_set = set(data_pool)
    
    missing_classes = list(all_classes - covered_classes)
    random.shuffle(missing_classes) 

    while missing_classes:
        target_cls = missing_classes.pop(0)
        if target_cls in covered_classes:
            continue 
            
        candidates = [img for img in class_to_images[target_cls] if img in pool_set and img not in selected_set]
        
        if not candidates:
            continue
            
        best_candidate = None
        best_gain = -1
        
        for cand in candidates:
            cand_classes = set(cand[2])
            gain = len(cand_classes - covered_classes)
            if gain > best_gain:
                best_gain = gain
                best_candidate = cand
            elif gain == best_gain:
                if random.random() < 0.5:
                    best_candidate = cand
        
        if best_candidate:
            selected_set.add(best_candidate)
            covered_classes.update(best_candidate[2])
            missing_classes = list(all_classes - covered_classes)
            random.shuffle(missing_classes)

    current_count = len(selected_set)
    if current_count < target_count:
        remaining_pool = [x for x in data_pool if x not in selected_set]
        needed = target_count - current_count
        random.shuffle(remaining_pool)
        selected_set.update(remaining_pool[:needed])
        
    return list(selected_set)

def main_generator():
    if os.path.exists(OUTPUT_ROOT):
        print(f"⚠️ Menghapus folder output lama: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT)

    print("⏳ Memindai dataset...")
    images = glob(os.path.join(SOURCE_ROOT, 'images', '*.jpg')) + \
             glob(os.path.join(SOURCE_ROOT, 'images', '*.jpeg')) + \
             glob(os.path.join(SOURCE_ROOT, 'images', '*.png'))
    
    data_map = [] 
    class_to_images = defaultdict(list)
    all_classes_found = set()

    for img in images:
        basename = os.path.basename(img)
        lbl_name = basename.rsplit('.', 1)[0] + '.txt'
        lbl_path = os.path.join(SOURCE_ROOT, 'labels', lbl_name)
        if os.path.exists(lbl_path):
            cls_ids = get_image_classes(lbl_path)
            if cls_ids:
                entry = (img, lbl_path, tuple(cls_ids))
                data_map.append(entry)
                all_classes_found.update(cls_ids)
                for c in cls_ids:
                    class_to_images[c].append(entry)

    total_imgs = len(data_map)
    print(f"✅ Total Data Valid: {total_imgs}")
    print(f"✅ Total Kelas Terdeteksi: {len(all_classes_found)}")

    # GLOBAL SPLIT
    n_test = int(total_imgs * TEST_SIZE)
    n_val = int(total_imgs * VAL_SIZE)
    print(f"🎯 Target Split -> Test: {n_test}, Val: {n_val}, Train: {total_imgs - n_test - n_val}")

    # A. Test Set
    test_list = greedy_stratified_select(data_map, class_to_images, n_test, all_classes_found)
    test_set_tuples = set(test_list)
    
    # B. Val Set
    remaining_after_test = [x for x in data_map if x not in test_set_tuples]
    val_list = greedy_stratified_select(remaining_after_test, class_to_images, n_val, all_classes_found)
    val_set_tuples = set(val_list)

    # C. Train Pool Master
    train_master_pool = [x for x in remaining_after_test if x not in val_set_tuples]

    print(f"🔒 Final Split -> Test: {len(test_list)}, Val: {len(val_list)}, Train Pool: {len(train_master_pool)}")
    
    # SCENARIOS
    for ratio in RATIOS:
        labeled_pct = int(ratio * 100)
        unlabeled_pct = 100 - labeled_pct
        scenario_name = f"scenario_L{labeled_pct}_U{unlabeled_pct}"
        print(f"\n🚀 Membuat {scenario_name}...")
        
        n_train_pool = len(train_master_pool)
        n_labeled_target = int(n_train_pool * ratio)
        if n_labeled_target == 0 and ratio > 0: n_labeled_target = 1
        
        # Greedy Selection untuk Labeled Data dari Pool
        pool_class_to_images = defaultdict(list)
        for entry in train_master_pool:
            img, lbl, c_ids = entry
            for c in c_ids:
                pool_class_to_images[c].append(entry)
                
        labeled_list = greedy_stratified_select(train_master_pool, pool_class_to_images, n_labeled_target, all_classes_found)
        labeled_set_tuples = set(labeled_list)
        
        unlabeled_list = [x for x in train_master_pool if x not in labeled_set_tuples]
        
        scen_path = os.path.join(OUTPUT_ROOT, scenario_name)

        def copy_files(dataset, subset, include_txt=True):
            img_dest = os.path.join(scen_path, subset, 'images')
            lbl_dest = os.path.join(scen_path, subset, 'labels')
            os.makedirs(img_dest, exist_ok=True)
            if include_txt: os.makedirs(lbl_dest, exist_ok=True)
            for img_src, lbl_src, _ in dataset:
                shutil.copy(img_src, img_dest)
                if include_txt:
                    shutil.copy(lbl_src, lbl_dest)

        copy_files(test_list, 'test', True)
        copy_files(val_list, 'val', True)
        copy_files(labeled_list, 'train_labeled', True)
        copy_files(unlabeled_list, 'train_unlabeled', False)

        datasets_dict = {
            'test': test_list,
            'val': val_list,
            'train_labeled': labeled_list,
            'train_unlabeled': unlabeled_list
        }
        
        save_statistics_detailed(scen_path, scenario_name, datasets_dict)

        abs_scen_path = os.path.abspath(scen_path)
        yaml_data = {
            'path': abs_scen_path,
            'train': os.path.join(abs_scen_path, 'train_labeled', 'images'),
            'val': os.path.join(abs_scen_path, 'val', 'images'),
            'test': os.path.join(abs_scen_path, 'test', 'images'),
            'nc': len(CLASS_NAMES_MAP),
            'names': CLASS_NAMES_MAP
        }
        with open(os.path.join(scen_path, 'data.yaml'), 'w') as f:
            yaml.dump(yaml_data, f, sort_keys=False)

    print(f"\n✅ DATASET SIAP! Folder output: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main_generator()