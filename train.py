import os
import shutil
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
from roboflow import Roboflow

def main():
    # 1. Bersihkan sisa folder lama agar tidak ada zip yang rusak
    print("Membersihkan folder lama...")
    for name in os.listdir("."):
        if name.startswith("baru-lagi"):
            if os.path.isdir(name):
                shutil.rmtree(name)
                print(f"Folder dihapus: {name}")
            elif os.path.isfile(name):
                os.remove(name)
                print(f"Berkas dihapus: {name}")

    # 2. Inisialisasi Roboflow dan download dataset otomatis
    print("Menghubungkan ke Roboflow...")
    rf = Roboflow(api_key="GcjsgvWWLCDmts88LhHy")
    project_name = "baru-lagi"
    
    project = rf.workspace("cicangoding").project(project_name)
    
    # Deteksi versi terbaru otomatis
    available_versions = [v.version for v in project.versions()]
    if not available_versions:
        raise RuntimeError("Tidak ada versi aktif yang ditemukan pada project ini.")
    
    latest_version = max(available_versions)
    print(f"Daftar versi aktif di Roboflow: {available_versions}")
    print(f"Menggunakan versi terbaru: {latest_version}")
    
    version = project.version(latest_version)
    dataset = version.download("yolov8")
    
    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    print(f"Dataset berhasil diunduh ke: {dataset.location}")
    print(f"Data YAML path: {data_yaml_path}")

    # 3. Inisialisasi Model YOLOv12s (Small)
    model_name = "yolo12s.pt"  
    print(f"Menginisialisasi model: {model_name}")
    model = YOLO(model_name)

    import torch
    device_to_use = 0 if torch.cuda.is_available() else "cpu"
    print(f"Menggunakan device: {device_to_use} (CUDA tersedia: {torch.cuda.is_available()})")

    # 4. Konfigurasi Latih Ringan & Cepat (Maksimal 100 Epoch)
    print("Memulai proses training...")
    results = model.train(
        data=data_yaml_path,
        epochs=100,             # Batasan maksimal 100 epoch agar tidak terlalu berat
        imgsz=640,              # Resolusi 640 standar (sangat cepat, ringan, bebas OOM)
        batch=16,               # Batch size 16 agar training stabil dan cepat selesai
        device=device_to_use,   
        
        # Pengaturan Standar Optimal (Disesuaikan untuk AdamW agar tidak overfit):
        optimizer="AdamW",      
        lr0=0.001,              # Diturunkan ke 0.001 agar AdamW stabil dan tidak merusak bobot pretrained
        lrf=0.01,               
        cos_lr=True,            
        patience=20,            # Early stopping jika tidak berkembang dalam 20 epoch
        freeze=10,              # Membekukan 10 layer pertama agar stabil pada data sedikit
        
        val=True                
    )
    
    print("Training Selesai! Hasil disimpan di folder runs/detect/train")

if __name__ == "__main__":
    main()
