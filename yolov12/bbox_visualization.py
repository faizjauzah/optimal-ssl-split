import cv2
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import tkinter as tk
from tkinter import filedialog

# ==========================================
# KONFIGURASI UTAMA (UBAH DISINI)
# ==========================================
# Ganti mode ke "manual" untuk memilih file via pop-up, atau "random" untuk acak
MODE = "manual"  

NUM_SAMPLES = 3  # Hanya berlaku jika MODE = "random"
FOLDER_IMAGES = "datasetGramasi/all_img/images"
FOLDER_LABELS = "datasetGramasi/all_img/labels"
# ==========================================

# --- DATA CLASS & WARNA ---
class_names = {
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

# --- DATA CLASS & WARNA ---
# Warna dikelompokkan berdasarkan jenis produk, dengan sedikit perbedaan kecerahan untuk membedakan gramasi (berat).
class_colors = {
    # --- DANCOW 1+ MADU (Shades of Gold/Yellow) ---
    "DANCOW 1+ Madu Advn ExcNutr 12x1000g N1 ID_REDESIGN3": (255, 215, 0),
    "DANCOW 1+ Madu Advn ExcNutr 12x800g ID_REDESIGN3": (235, 200, 0),
    "DANCOW 1+ Madu Advn ExcNutr 24x400g N1 ID_REDESIGN3": (215, 180, 0),
    "DANCOW 1+ Madu Advn ExcNutr 40x200g ID_REDESIGN3": (195, 160, 0),

    # --- DANCOW 1+ VAN (Shades of Light Orange) ---
    "DANCOW 1+ Van Advn ExcNutr 12x1000g ID_REDESIGN3": (255, 170, 0),
    "DANCOW 1+ Van Advn ExcNutr 12x800g N1 ID_REDESIGN3": (240, 150, 0),
    "DANCOW 1+ Van Advn ExcNutr 24x400g ID_REDESIGN3": (220, 130, 0),
    "DANCOW 1+ Van Advn ExcNutr 40x200g N1 ID_REDESIGN3": (200, 110, 0),

    # --- DANCOW 3+ MADU (Shades of Dark Orange) ---
    "DANCOW 3+ Madu Advn ExcNutr 12x1000g ID_REDESIGN3": (255, 100, 0),
    "DANCOW 3+ Madu Advn ExcNutr 12x800g ID_REDESIGN3": (230, 85, 0),
    "DANCOW 3+ Madu Advn ExcNutr 24x400g ID_REDESIGN3": (205, 70, 0),

    # --- DANCOW 3+ VAN (Shades of Red-Orange) ---
    "DANCOW 3+ Van Advn ExcNutr 12x1000g ID_REDESIGN3": (255, 69, 0),
    "DANCOW 3+ Van Advn ExcNutr 12x800g ID_REDESIGN3": (225, 50, 0),
    "DANCOW 3+ Van Advn ExcNutr 24x400g ID_REDESIGN3": (195, 35, 0),

    # --- DANCOW 5+ MADU (Shades of Dark Red) ---
    "DANCOW 5+ Madu Advn ExcNutr 12x1000g ID_REDESIGN3": (200, 50, 0),
    "DANCOW 5+ Madu Advn ExcNutr 12x800g ID_REDESIGN3": (170, 30, 0),
    "DANCOW 5+ Madu Advn ExcNutr 24x400g ID_REDESIGN3": (140, 15, 0),

    # --- LACTOGEN 1 (Shades of Lime/Bright Green) ---
    "LACTOGEN 1 Happynutri 12x1kg ID_REDESIGN2": (50, 205, 50),
    "LACTOGEN 1 Happynutri 12x750g N1 ID_REDESIGN2": (45, 185, 45),
    "LACTOGEN 1 Happynutri 24x350g N1 ID_REDESIGN2": (40, 165, 40),
    "LACTOGEN 1 Happynutri 40x180g N1 ID_REDESIGN2": (35, 145, 35),

    # --- LACTOGEN 2 (Shades of Forest Green) ---
    "LACTOGEN 2 Happynutri 12x1kg ID_REDESIGN2": (34, 139, 34),
    "LACTOGEN 2 Happynutri 12x750g N1 ID_REDESIGN2": (28, 120, 28),
    "LACTOGEN 2 Happynutri 24x350g N1 ID_REDESIGN2": (22, 100, 22),
    "LACTOGEN 2 Happynutri 40x180g N1 ID_REDESIGN2": (16, 80, 16),

    # --- LACTOGEN PRO 0-6 (Shades of Teal/Sea Green) ---
    "LACTOGEN PRO 0-6 mo 12x1kg ID": (0, 150, 100),
    "LACTOGEN PRO 0-6 mo 12x735g ID": (0, 130, 85),
    "LACTOGEN PRO 0-6 mo 24x350g ID": (0, 110, 70),
    "LACTOGEN PRO 0-6 mo 40x180g ID": (0, 90, 55),

    # --- LACTOGEN PRO 6-12 (Shades of Very Dark Green) ---
    "LACTOGEN PRO 6-12 mo 12x1kg ID": (0, 140, 0),
    "LACTOGEN PRO 6-12 mo 12x735g ID": (0, 120, 0),
    "LACTOGEN PRO 6-12 mo 24x350g ID": (0, 100, 0),
    "LACTOGEN PRO 6-12 mo 40x180g ID": (0, 80, 0),

    # --- LACTOGROW 3 HONEY (Shades of Sky/Light Blue) ---
    "LACTOGROW 3 Happynutri Honey 12x1kg N1 ID_REDESIGN2": (0, 191, 255),
    "LACTOGROW 3 Happynutri Honey 12x750g N1 ID_REDESIGN2": (0, 170, 230),
    "LACTOGROW 3 Happynutri Honey 24x350g N1 ID_REDESIGN2": (0, 150, 205),
    "LACTOGROW 3 Happynutri Honey 40x180g N1 ID_REDESIGN2": (0, 130, 180),

    # --- LACTOGROW 3 VAN (Shades of Dodger Blue) ---
    "LACTOGROW 3 Happynutri Van 12x1kg ID_REDESIGN2": (30, 144, 255),
    "LACTOGROW 3 Happynutri Van 12x750g N1 ID_REDESIGN2": (25, 120, 225),
    "LACTOGROW 3 Happynutri Van 24x350g N1 ID_REDESIGN2": (20, 100, 195),

    # --- LACTOGROW PRO 1+ HONEY (Shades of Standard Blue) ---
    "LACTOGROW PRO 1+ Honey 12x1kg ID": (0, 0, 255),
    "LACTOGROW PRO 1+ Honey 12x735g ID": (0, 0, 220),
    "LACTOGROW PRO 1+ Honey 24x350g ID": (0, 0, 185),
    "LACTOGROW PRO 1+ Honey 40x145g ID": (0, 0, 150),

    # --- LACTOGROW PRO 1+ VAN (Shades of Indigo/Purple) ---
    "LACTOGROW PRO 1+ Van 12x1kg ID": (75, 0, 130),
    "LACTOGROW PRO 1+ Van 12x735g ID": (105, 0, 160),
    "LACTOGROW PRO 1+ Van 24x350g ID": (135, 0, 190)
}

def plot_box_better(img, bbox, label, color, line_thickness=3):
    x1, y1, x2, y2 = bbox
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    alpha = 0.25  
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

    font_scale = 0.6
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    text_x = x1
    text_y = y1 - 10 
    if text_y - text_h < 0: 
        text_y = y1 + text_h + 10 

    p = 5 
    cv2.rectangle(img, (text_x - p, text_y - text_h - p), (text_x + text_w + p, text_y + p), color, -1)
    
    text_color = (0, 0, 0)
    if sum(color) < 300: 
        text_color = (255, 255, 255)

    cv2.putText(img, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img

def select_file_gui():
    """Membuka pop-up window untuk memilih file secara manual."""
    root = tk.Tk()
    root.withdraw() # Sembunyikan window utama yang kosong
    
    # Ambil path absolute agar pop-up langsung membuka folder yang benar
    initial_dir = os.path.abspath(FOLDER_IMAGES) 
    
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Pilih Gambar untuk Divisualisasi",
        filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
    )
    return file_path

def preview_dataset():
    if not os.path.exists(FOLDER_IMAGES):
        print(f"Error: Folder {FOLDER_IMAGES} tidak ditemukan.")
        return

    # --- LOGIKA PEMILIHAN FILE ---
    samples = []
    if MODE == "manual":
        print("Silakan pilih gambar melalui jendela pop-up...")
        selected_file = select_file_gui()
        if not selected_file:
            print("Pemilihan file dibatalkan.")
            return
        
        # Ekstrak nama file dari path lengkap
        samples.append(os.path.basename(selected_file))
        actual_samples = 1
        
    else: # Mode Random
        all_images = [f for f in os.listdir(FOLDER_IMAGES) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not all_images:
            print("Tidak ada gambar di folder.")
            return
        actual_samples = min(NUM_SAMPLES, len(all_images))
        samples = random.sample(all_images, actual_samples)

    # --- LOGIKA GRID DINAMIS ---
    cols = 3 if actual_samples >= 3 else actual_samples
    rows = (actual_samples + cols - 1) // cols
    
    # Sesuaikan ukuran kanvas jika hanya 1 gambar
    if actual_samples == 1:
        fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(20, 7 * rows))
    
    if actual_samples == 1:
        axes = [axes] 
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten() 
    
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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img)
        axes[i].set_title(img_name, fontsize=12, fontweight='bold')
        axes[i].axis("off")

    for j in range(len(samples), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

# RUN PROGRAM
if __name__ == "__main__":
    preview_dataset()