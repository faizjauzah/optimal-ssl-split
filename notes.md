# Ini adalah catatan dari project skripsi saya

saya akan menggunakan git untuk version handling dari project ini.

berikut link dari git:

## Virtual Environment

untuk virtual environment dari project ini saya menggunakan conda.

untuk mengaktifkan env ketika ingin mengerjakan project ketik `conda activate yolov12`.

disitu sudah terinstall semua dependencies yang akan digunakan, untuk lebih lengkapnya ketik `conda list`.

ketika sudah selesai mengerjakan project, matikan env dengan `conda deactivate`.

## Struktur Project

struktur awal dari project ini ada 4 folder utama dan 1 file notes. 4 folder tersebut adalah folder dataset, datasetGramasi, hasilAkhir dan yolov12.

saya akan jelaskan masing-masing dari folder tersebut.

### folder datasetGramasi

folder ini adalah folder awal yang saya dapatkan dari dosen pembimbing saya. isinya awalnya cuma grouping new, split_v1, split_v2. dikarenakan saya butuh seluruh gambar dan label dijadikan satu, maka saya buat folder baru `all_img` yang isinya gabungan keseluruhan gambar dan label. saya juga memberikan tambahan statistik dari dataset di folder tersebut.

### folder dataset

folder ini adalah folder hasil program data splitting yang akan mengsplit folder `all_img` menjadi beberapa skenario data label dan data unlabel (10:90, 20:80, ..., 90:10).

### folder hasilAkhir

folder ini adalah folder yang isinya hasil trainingan dari masing-masing skenario dan juga iterasinya.

### folder yolov12

ini adalah folder terpenting yang isinya program-program utama dan juga model yolov12 yang akan digunakan. semua kode program akan ada disini.

nama-nama kode program yang ada di folder ini:

- `data_splitting.py`

  menggunakan metode greedy stratified sampling untuk memastikan bahwa di setiap folder (khususnya folder test dan val) terdapat minimal 1 kelas. agar performa hasil training dapat maksimal.

- `test_dataset_validation.py`

  untuk memvalidasi di setiap folder test kelasnya lengkap.

- `visualisasi_bbox.py`

  untuk melihat secara sekilas bounding box yang ada.

### dibawah ini merupakan pipeline dari project

1. `train_teacher_L<scenario>.py`

   contoh nama `train_teacher_L10.py` berarti ini merupakan kode untuk skenario data label 10%. ini merupakan langkah pertama dari pipeline project ini yaitu melatih model teacher dari data berlabel yang ada. hasil dari trainingan ini akan masuk ke folder `hasilAkhir`.

2. `generate_pseudo_labels_L<scenario>.py`

   hasil training model teacher tadi akan digunakan untuk memprediksi labels ke data unlabel sisanya. misal di skenario 10% data label maka kode ini akan digunakan untuk memprediksi pseudo labels dari 90% data unlabel yang ada. hasilnya adalah folder labels yang isinya file .txt.

3. `create_student_config_L<scenario>.py`

   digunakan untuk menggabunggan data label dan data pseudo label menjadi satu di `data_student.yaml`.

4. `train_student_L<scenario>.py`

   melatih model student dengan melakukan pelatihan lagi menggunakan gabungan data label dan pseudo label. hasil dari trainingan ini akan masuk ke folder `hasilAkhir`. ini adalah akhir dari pipeline project ini. hasil dari model student ini nantinya akan dibandingan dengan skenario-skenario lainnya sehigga mendapatkan rasio data label : data unlabel yang terbaik untuk semi-supervised learning dalam ruang lingkup ritel.
