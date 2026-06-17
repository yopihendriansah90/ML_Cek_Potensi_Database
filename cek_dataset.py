import pandas as pd

# Membaca dataset
df = pd.read_csv("dataset/diabetes_data_upload.csv")

# Menampilkan 5 data pertama
print("5 Data Pertama:")
print(df.head())

# Menampilkan ukuran dataset
print("\nUkuran Dataset:")
print(df.shape)

# Menampilkan nama kolom
print("\nNama Kolom:")
print(df.columns)

# Menampilkan informasi dataset
print("\nInformasi Dataset:")
print(df.info())

# Mengecek missing value
print("\nJumlah Missing Value:")
print(df.isnull().sum())

# Mengecek jumlah class
print("\nDistribusi Class:")
print(df["class"].value_counts())