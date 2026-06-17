import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =========================
# 1. Folder Output
# =========================
MODEL_DIR = Path("model")
OUTPUT_DIR = Path("output")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# 2. Load Dataset Final
# =========================
df = pd.read_csv("dataset/diabetes_clean_dedup.csv")

print("===== DATASET FINAL =====")
print("Ukuran dataset:", df.shape)
print("\nDistribusi class:")
print(df["class"].value_counts())


# =========================
# 3. Pisahkan Fitur dan Target
# =========================
X = df.drop("class", axis=1)
y = df["class"]

feature_columns = X.columns.tolist()

print("\nJumlah fitur:", len(feature_columns))
print("Nama fitur:")
print(feature_columns)


# =========================
# 4. Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===== SPLIT DATA =====")
print("Data training:", X_train.shape)
print("Data testing:", X_test.shape)

print("\nDistribusi class pada data training:")
print(y_train.value_counts())

print("\nDistribusi class pada data testing:")
print(y_test.value_counts())


# =========================
# 5. Definisi Model
# =========================
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ]),

    "Naive Bayes": GaussianNB()
}


# =========================
# 6. Training dan Evaluasi
# =========================
results = []

best_model_name = None
best_model = None
best_f1 = 0

print("\n===== HASIL EVALUASI MODEL =====")

for model_name, model in models.items():
    print(f"\n--- {model_name} ---")

    # Training
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # Pilih model terbaik berdasarkan F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = model_name
        best_model = model


# =========================
# 7. Simpan Hasil Evaluasi
# =========================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-score", ascending=False)

print("\n===== PERBANDINGAN MODEL =====")
print(results_df)

results_df.to_csv("output/model_comparison.csv", index=False)


# =========================
# 8. Simpan Model Terbaik
# =========================
joblib.dump(best_model, "model/best_model.pkl")
joblib.dump(feature_columns, "model/feature_columns.pkl")

print("\n===== MODEL TERBAIK =====")
print("Model terbaik:", best_model_name)
print("F1-score terbaik:", round(best_f1, 4))

print("\nModel berhasil disimpan ke:")
print("model/best_model.pkl")

print("\nFeature columns berhasil disimpan ke:")
print("model/feature_columns.pkl")


# =========================
# 9. Visualisasi Perbandingan Model
# =========================
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["F1-score"])
plt.title("Perbandingan F1-score Model Klasifikasi")
plt.xlabel("Model")
plt.ylabel("F1-score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("output/perbandingan_model_f1_score.png")
plt.close()

print("\nGrafik perbandingan model disimpan ke:")
print("output/perbandingan_model_f1_score.png")