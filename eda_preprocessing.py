import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. Path Dataset
# =========================
DATA_PATH = "dataset/diabetes_data_upload.csv"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# 2. Load Dataset
# =========================
df = pd.read_csv(DATA_PATH)

print("===== DATASET AWAL =====")
print(df.head())
print("\nUkuran dataset:", df.shape)

# =========================
# 3. Rapikan Nama Kolom
# =========================
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\n===== NAMA KOLOM SETELAH DIRAPIKAN =====")
print(df.columns)

# =========================
# 4. Cek Missing Value
# =========================
print("\n===== MISSING VALUE =====")
print(df.isnull().sum())

# =========================
# 5. Cek Data Duplikat
# =========================
duplicate_count = df.duplicated().sum()
print("\n===== DATA DUPLIKAT =====")
print("Jumlah data duplikat:", duplicate_count)

# =========================
# 6. Cek Nilai Unik Setiap Kolom Kategorikal
# =========================
print("\n===== NILAI UNIK SETIAP KOLOM =====")
for col in df.columns:
    print(f"{col}: {df[col].unique()}")

# =========================
# 7. Distribusi Class
# =========================
print("\n===== DISTRIBUSI CLASS =====")
print(df["class"].value_counts())

plt.figure(figsize=(6, 4))
df["class"].value_counts().plot(kind="bar")
plt.title("Distribusi Class Diabetes")
plt.xlabel("Class")
plt.ylabel("Jumlah Data")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distribusi_class.png")
plt.close()

# =========================
# 8. Distribusi Gender Berdasarkan Class
# =========================
gender_class = pd.crosstab(df["gender"], df["class"])

print("\n===== DISTRIBUSI GENDER BERDASARKAN CLASS =====")
print(gender_class)

gender_class.plot(kind="bar", figsize=(7, 4))
plt.title("Distribusi Gender Berdasarkan Class")
plt.xlabel("Gender")
plt.ylabel("Jumlah Data")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gender_berdasarkan_class.png")
plt.close()

# =========================
# 9. Encoding Data Kategorikal
# =========================
df_encoded = df.copy()

# Encoding target class
df_encoded["class"] = df_encoded["class"].map({
    "Positive": 1,
    "Negative": 0
})

# Encoding gender
df_encoded["gender"] = df_encoded["gender"].map({
    "Male": 1,
    "Female": 0
})

# Encoding semua kolom Yes/No
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = df_encoded[col].map({
            "Yes": 1,
            "No": 0
        })

print("\n===== DATA SETELAH ENCODING =====")
print(df_encoded.head())

print("\n===== TIPE DATA SETELAH ENCODING =====")
print(df_encoded.info())

# =========================
# 10. Simpan Dataset Bersih
# =========================
df_encoded.to_csv("dataset/diabetes_clean.csv", index=False)

print("\nDataset bersih berhasil disimpan ke:")
print("dataset/diabetes_clean.csv")

print("\nGrafik berhasil disimpan ke folder:")
print("output/")