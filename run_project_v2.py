"""
CLI Runner Project Data Mining
Prediksi Risiko Diabetes Menggunakan Algoritma Klasifikasi

Versi: 2.0
Perbaikan:
1. Memperbaiki error PosixPath.__format__ pada menu cek kelengkapan file.
2. Menambahkan menu untuk menjalankan semua proses dari awal sampai Streamlit.
3. Menambahkan pembuatan folder dataset, model, dan output jika belum ada.

Cara pakai:
1. Simpan file ini di root folder project, sejajar dengan:
   - cek_dataset.py
   - eda_preprocessing.py
   - handle_duplicates.py
   - train_model.py
   - app.py

2. Aktifkan virtual environment:
   Linux/Mac:
      source venv/bin/activate

   Windows:
      venv\\Scripts\\activate

3. Jalankan:
   Linux/Mac:
      python3 run_project.py

   Windows:
      python run_project.py
"""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    "cek_dataset": PROJECT_ROOT / "cek_dataset.py",
    "eda_preprocessing": PROJECT_ROOT / "eda_preprocessing.py",
    "handle_duplicates": PROJECT_ROOT / "handle_duplicates.py",
    "train_model": PROJECT_ROOT / "train_model.py",
    "app": PROJECT_ROOT / "app.py",
}

REQUIRED_PATHS = {
    "dataset_asli": PROJECT_ROOT / "dataset" / "diabetes_data_upload.csv",
    "dataset_clean": PROJECT_ROOT / "dataset" / "diabetes_clean.csv",
    "dataset_dedup": PROJECT_ROOT / "dataset" / "diabetes_clean_dedup.csv",
    "best_model": PROJECT_ROOT / "model" / "best_model.pkl",
    "feature_columns": PROJECT_ROOT / "model" / "feature_columns.pkl",
}


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def pause() -> None:
    input("\nTekan ENTER untuk kembali ke menu...")


def ensure_folders() -> None:
    """Membuat folder dasar project jika belum ada."""
    for folder_name in ["dataset", "model", "output"]:
        (PROJECT_ROOT / folder_name).mkdir(exist_ok=True)


def run_python_script(script_path: Path) -> bool:
    """Menjalankan script Python menggunakan interpreter dari environment aktif."""
    if not script_path.exists():
        print(f"\n[ERROR] File tidak ditemukan: {script_path.name}")
        return False

    print_header(f"Menjalankan {script_path.name}")
    print(f"Lokasi file: {script_path}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=False
        )

        if result.returncode == 0:
            print(f"\n[SUKSES] {script_path.name} selesai dijalankan.")
            return True

        print(f"\n[GAGAL] {script_path.name} berhenti dengan kode error: {result.returncode}")
        return False

    except KeyboardInterrupt:
        print("\n\n[DIHENTIKAN] Proses dihentikan oleh pengguna.")
        return False

    except Exception as error:
        print(f"\n[ERROR] Terjadi kesalahan saat menjalankan {script_path.name}:")
        print(error)
        return False


def run_streamlit_app() -> None:
    """Menjalankan aplikasi Streamlit."""
    app_path = SCRIPTS["app"]

    if not app_path.exists():
        print("\n[ERROR] File app.py tidak ditemukan.")
        pause()
        return

    if not REQUIRED_PATHS["best_model"].exists() or not REQUIRED_PATHS["feature_columns"].exists():
        print("\n[ERROR] File model belum tersedia.")
        print("Pastikan file berikut sudah ada:")
        print("- model/best_model.pkl")
        print("- model/feature_columns.pkl")
        print("\nJalankan menu 4 untuk training model, atau menu 5 untuk full pipeline.")
        pause()
        return

    print_header("Menjalankan Aplikasi Streamlit")
    print("Aplikasi akan berjalan di browser.")
    print("Jika browser tidak terbuka otomatis, buka alamat:")
    print("http://localhost:8501")
    print("\nUntuk menghentikan aplikasi, tekan CTRL + C di terminal.")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=PROJECT_ROOT,
            check=False
        )

    except KeyboardInterrupt:
        print("\n\n[DIHENTIKAN] Aplikasi Streamlit dihentikan.")

    except Exception as error:
        print("\n[ERROR] Gagal menjalankan Streamlit.")
        print("Pastikan Streamlit sudah terinstall:")
        print("pip install streamlit")
        print("\nDetail error:")
        print(error)

    pause()


def check_project_files() -> None:
    """Mengecek kelengkapan file dan folder project."""
    ensure_folders()

    print_header("Cek Kelengkapan File Project")

    print("\n[Script Python]")
    for name, path in SCRIPTS.items():
        status = "ADA" if path.exists() else "TIDAK ADA"
        print(f"- {path.name:25} : {status}")

    print("\n[Dataset dan Model]")
    for name, path in REQUIRED_PATHS.items():
        status = "ADA" if path.exists() else "TIDAK ADA"

        # Perbaikan utama:
        # PosixPath tidak bisa langsung diberi format lebar seperti :35.
        # Karena itu path relatif diubah dulu menjadi string.
        relative_path = str(path.relative_to(PROJECT_ROOT))
        print(f"- {relative_path:35} : {status}")

    print("\n[Folder]")
    for folder_name in ["dataset", "model", "output"]:
        folder_path = PROJECT_ROOT / folder_name
        status = "ADA" if folder_path.exists() else "TIDAK ADA"
        print(f"- {folder_name:25} : {status}")

    pause()


def run_full_pipeline() -> bool:
    """Menjalankan seluruh proses dari cek dataset sampai training model."""
    ensure_folders()

    print_header("Menjalankan Full Pipeline")

    if not REQUIRED_PATHS["dataset_asli"].exists():
        print("\n[ERROR] Dataset asli belum ditemukan.")
        print("Pastikan file dataset ada di:")
        print("dataset/diabetes_data_upload.csv")
        return False

    steps = [
        ("Cek Dataset", SCRIPTS["cek_dataset"]),
        ("EDA dan Preprocessing", SCRIPTS["eda_preprocessing"]),
        ("Hapus Duplikat", SCRIPTS["handle_duplicates"]),
        ("Training Model", SCRIPTS["train_model"]),
    ]

    for step_name, script_path in steps:
        print_header(f"STEP: {step_name}")
        success = run_python_script(script_path)

        if not success:
            print(f"\nPipeline dihentikan karena step '{step_name}' gagal.")
            return False

    print_header("FULL PIPELINE SELESAI")
    print("Semua tahap berhasil dijalankan.")
    print("\nFile yang seharusnya sudah terbentuk:")
    print("- dataset/diabetes_clean.csv")
    print("- dataset/diabetes_clean_dedup.csv")
    print("- model/best_model.pkl")
    print("- model/feature_columns.pkl")
    print("- output/model_comparison.csv")
    print("- output/perbandingan_model_f1_score.png")

    return True


def run_full_pipeline_with_streamlit() -> None:
    """Menjalankan semua proses dari awal sampai aplikasi Streamlit."""
    print_header("Menjalankan Semua Proses dari Awal sampai Streamlit")

    success = run_full_pipeline()

    if not success:
        print("\n[GAGAL] Aplikasi Streamlit tidak dijalankan karena pipeline belum berhasil.")
        pause()
        return

    print("\n[SUKSES] Pipeline selesai. Sekarang aplikasi Streamlit akan dijalankan.")
    print("Catatan: Setelah Streamlit berjalan, proses akan tetap aktif sampai dihentikan dengan CTRL + C.")

    run_streamlit_app()


def show_menu() -> None:
    print_header("CLI RUNNER PROJECT DATA MINING DIABETES")
    print("Pilih menu yang ingin dijalankan:\n")
    print("1. Cek dataset")
    print("2. EDA dan preprocessing")
    print("3. Hapus data duplikat")
    print("4. Training dan evaluasi model")
    print("5. Jalankan full pipeline")
    print("6. Jalankan aplikasi Streamlit")
    print("7. Cek kelengkapan file project")
    print("8. Jalankan semua dari awal sampai Streamlit")
    print("0. Keluar")


def main() -> None:
    ensure_folders()

    while True:
        show_menu()
        choice = input("\nMasukkan pilihan menu: ").strip()

        if choice == "1":
            run_python_script(SCRIPTS["cek_dataset"])
            pause()

        elif choice == "2":
            run_python_script(SCRIPTS["eda_preprocessing"])
            pause()

        elif choice == "3":
            run_python_script(SCRIPTS["handle_duplicates"])
            pause()

        elif choice == "4":
            run_python_script(SCRIPTS["train_model"])
            pause()

        elif choice == "5":
            run_full_pipeline()
            pause()

        elif choice == "6":
            run_streamlit_app()

        elif choice == "7":
            check_project_files()

        elif choice == "8":
            run_full_pipeline_with_streamlit()

        elif choice == "0":
            print("\nKeluar dari CLI Runner. Selesai.")
            break

        else:
            print("\n[WARNING] Pilihan tidak valid. Silakan pilih menu 0 sampai 8.")
            pause()


if __name__ == "__main__":
    main()
