import pandas as pd

# Membaca dataset yang sudah diencoding
df = pd.read_csv("dataset/diabetes_clean.csv")

print("===== DATASET SEBELUM MENGHAPUS DUPLIKAT =====")
print("Ukuran dataset:", df.shape)

print("\nDistribusi class sebelum hapus duplikat:")
print(df["class"].value_counts())

# Cek jumlah duplikat
duplicate_count = df.duplicated().sum()
print("\nJumlah data duplikat:", duplicate_count)

# Menghapus data duplikat
df_dedup = df.drop_duplicates()

print("\n===== DATASET SETELAH MENGHAPUS DUPLIKAT =====")
print("Ukuran dataset:", df_dedup.shape)

print("\nDistribusi class setelah hapus duplikat:")
print(df_dedup["class"].value_counts())

# Simpan dataset bersih tanpa duplikat
df_dedup.to_csv("dataset/diabetes_clean_dedup.csv", index=False)

print("\nDataset tanpa duplikat berhasil disimpan ke:")
print("dataset/diabetes_clean_dedup.csv")