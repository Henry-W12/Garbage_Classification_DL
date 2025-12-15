Cara Menjalankan Program:

1. Clone Repository
git clone https://github.com/Henry/Garbage_Classification_DL.git

2. Masuk ke Folder Project
cd Garbage_Classification_DL

3. Install Dependency
Pastikan Python telah terinstall, kemudian jalankan:
pip install -r requirements.txt

4. Pastikan File Model Tersedia
Pastikan file model best_model.pth berada di dalam folder project sesuai dengan path yang digunakan pada kode program.
Catatan: Dataset tidak disertakan di dalam repository. Aplikasi ini menggunakan model yang telah dilatih sebelumnya.

5. Jalankan Aplikasi Streamlit
streamlit run app.py

6. Gunakan Aplikasi
Buka browser sesuai alamat yang ditampilkan oleh Streamlit
Upload gambar sampah
Sistem akan menampilkan hasil klasifikasi berdasarkan model ResNet50
