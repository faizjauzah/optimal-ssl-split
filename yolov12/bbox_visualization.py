import cv2
import matplotlib.pyplot as plt
import os
import random
import numpy as np

# ==========================================
# KONFIGURASI UTAMA (UBAH DISINI)
# ==========================================
NUM_SAMPLES = 3  # Ubah angka ini sesuai keinginan (contoh: 1, 6, 9)
FOLDER_IMAGES = "datasetGramasi/all_img/images"
FOLDER_LABELS = "datasetGramasi/all_img/labels"
# ==========================================


# --- DATA CLASS & WARNA ---
class_names = {
    0: "DANCOW 1+ MADU", 1: "DANCOW 1+ MADU", 2: "DANCOW 1+ MADU", 3: "DANCOW 1+ MADU",
    4: "DANCOW 1+ VAN", 5: "DANCOW 1+ VAN", 6: "DANCOW 1+ VAN", 7: "DANCOW 1+ VAN",
    8: "DANCOW 3+ MADU", 9: "DANCOW 3+ MADU", 10: "DANCOW 3+ MADU",
    11: "DANCOW 3+ VAN", 12: "DANCOW 3+ VAN", 13: "DANCOW 3+ VAN",
    14: "DANCOW 5+ MADU", 15: "DANCOW 5+ MADU", 16: "DANCOW 5+ MADU",
    17: "LACTOGEN 1", 18: "LACTOGEN 1", 19: "LACTOGEN 1", 20: "LACTOGEN 1",
    21: "LACTOGEN 2", 22: "LACTOGEN 2", 23: "LACTOGEN 2", 24: "LACTOGEN 2",
    25: "LACTOGEN PRO 0-6", 26: "LACTOGEN PRO 0-6",
    27: "LACTOGEN PRO 0-6", 28: "LACTOGEN PRO 0-6",
    29: "LACTOGEN PRO 6-12", 30: "LACTOGEN PRO 6-12",
    31: "LACTOGEN PRO 6-12", 32: "LACTOGEN PRO 6-12",
    33: "LACTOGROW 3 MADU", 34: "LACTOGROW 3 MADU",
    35: "LACTOGROW 3 MADU", 36: "LACTOGROW 3 MADU",
    37: "LACTOGROW 3 VAN", 38: "LACTOGROW 3 VAN", 39: "LACTOGROW 3 VAN",
    40: "LACTOGROW PRO 1+ MADU", 41: "LACTOGROW PRO 1+ MADU",
    42: "LACTOGROW PRO 1+ MADU", 43: "LACTOGROW PRO 1+ MADU",
    44: "LACTOGROW PRO 1+ VAN", 45: "LACTOGROW PRO 1+ VAN", 46: "LACTOGROW PRO 1+ VAN"
}

class_colors = {
    "DANCOW 1+ MADU": (255, 215, 0),      # Gold
    "DANCOW 1+ VAN": (255, 170, 0),       # Orange-ish
    "DANCOW 3+ MADU": (255, 100, 0),      # Darker Orange
    "DANCOW 3+ VAN": (255, 69, 0),        # Red-Orange
    "DANCOW 5+ MADU": (200, 50, 0),       # Dark Red
    "LACTOGEN 1": (50, 205, 50),          # Lime Green
    "LACTOGEN 2": (34, 139, 34),          # Forest Green
    "LACTOGEN PRO 0-6": (0, 100, 0),      # Dark Green
    "LACTOGEN PRO 6-12": (0, 80, 0),      # Very Dark Green
    "LACTOGROW 3 MADU": (0, 191, 255),    # Deep Sky Blue
    "LACTOGROW 3 VAN": (30, 144, 255),    # Dodger Blue
    "LACTOGROW PRO 1+ MADU": (0, 0, 255), # Blue
    "LACTOGROW PRO 1+ VAN": (0, 0, 139),  # Dark Blue
}

def plot_box_better(img, bbox, label, color, line_thickness=3):
    """
    Menggambar bounding box dengan tampilan modern:
    - Garis tebal & Overlay transparan
    - Label teks smart positioning
    """
    x1, y1, x2, y2 = bbox
    
    # 1. Gambar Overlay Transparan (Isi Kotak)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    alpha = 0.25  # Transparansi
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # 2. Gambar Border Kotak (Solid)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

    # 3. Siapkan Label Teks
    font_scale = 0.6
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Logika posisi label (agar tidak keluar gambar)
    text_x = x1
    text_y = y1 - 10 
    if text_y - text_h < 0: # Jika mentok atas
        text_y = y1 + text_h + 10 # Pindah ke dalam kotak

    # Background Label
    p = 5 
    cv2.rectangle(img, (text_x - p, text_y - text_h - p), (text_x + text_w + p, text_y + p), color, -1)
    
    # Warna Teks (Putih untuk bg gelap, Hitam untuk bg terang)
    text_color = (0, 0, 0)
    if sum(color) < 300: 
        text_color = (255, 255, 255)

    cv2.putText(img, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img

def preview_dataset(num_samples):
    # Cek folder
    if not os.path.exists(FOLDER_IMAGES):
        print(f"Error: Folder {FOLDER_IMAGES} tidak ditemukan.")
        return

    all_images = [f for f in os.listdir(FOLDER_IMAGES) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not all_images:
        print("Tidak ada gambar di folder.")
        return

    # Ambil sampel acak
    actual_samples = min(num_samples, len(all_images))
    samples = random.sample(all_images, actual_samples)

    # --- LOGIKA GRID DINAMIS ---
    # Jika samples < 3, kolom menyesuaikan jumlah sample. Max 3 kolom.
    cols = 3 if actual_samples >= 3 else actual_samples
    rows = (actual_samples + cols - 1) // cols
    
    # Buat figure
    # figsize disesuaikan: lebar 20, tinggi 7 per baris
    fig, axes = plt.subplots(rows, cols, figsize=(20, 7 * rows))
    
    # --- HANDLING AXES ARRAY ---
    # Matplotlib mengembalikan objek berbeda tergantung jumlah row/col
    # Kita paksa jadi list datar (flat list) agar mudah di-loop
    if actual_samples == 1:
        axes = [axes] # Bungkus jadi list jika cuma 1 objek
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten() # Ratakan jika array multi-dimensi
    
    # Loop gambar
    for i, img_name in enumerate(samples):
        img_path = os.path.join(FOLDER_IMAGES, img_name)
        label_path = os.path.join(FOLDER_LABELS, os.path.splitext(img_name)[0] + '.txt')

        img = cv2.imread(img_path)
        if img is None: continue
        
        h_img, w_img, _ = img.shape

        if os.path.exists(label_path):
            with open(label_path) as f:
                lines = f.readlines()
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    cls = int(parts[0])
                    xc, yc, bw, bh = parts[1], parts[2], parts[3], parts[4]

                    x1 = int((xc - bw / 2) * w_img)
                    y1 = int((yc - bh / 2) * h_img)
                    x2 = int((xc + bw / 2) * w_img)
                    y2 = int((yc + bh / 2) * h_img)

                    name = class_names.get(cls, "UNKNOWN")
                    color = class_colors.get(name, (200, 200, 200))

                    img = plot_box_better(img, (x1, y1, x2, y2), name, color)

        # Convert ke RGB untuk Matplotlib
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img)
        axes[i].set_title(img_name, fontsize=12, fontweight='bold')
        axes[i].axis("off")

    # Matikan axis untuk slot kosong (jika ada sisa grid)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

# RUN PROGRAM
if __name__ == "__main__":
    preview_dataset(NUM_SAMPLES)