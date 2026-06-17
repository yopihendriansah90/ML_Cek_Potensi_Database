import streamlit as st
import pandas as pd
import joblib


# =========================
# 1. Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="centered"
)


# =========================
# 2. Load Model dan Fitur
# =========================
model = joblib.load("model/best_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")


# =========================
# 3. Fungsi Encoding
# =========================
def yes_no_to_number(value):
    return 1 if value == "Yes" else 0


def gender_to_number(value):
    return 1 if value == "Male" else 0


# =========================
# 4. Tampilan Aplikasi
# =========================
st.title("🩺 Aplikasi Prediksi Risiko Diabetes")
st.write(
    "Aplikasi ini menggunakan model klasifikasi Random Forest "
    "untuk memprediksi risiko diabetes berdasarkan gejala awal pasien."
)

st.warning(
    "Catatan: Aplikasi ini dibuat untuk keperluan akademik/praktik Data Mining, "
    "bukan sebagai alat diagnosis medis resmi."
)

st.divider()


# =========================
# 5. Form Input Pengguna
# =========================
st.subheader("Masukkan Data Pasien")

age = st.number_input(
    "Usia",
    min_value=1,
    max_value=120,
    value=40
)

gender = st.selectbox(
    "Jenis Kelamin",
    ["Male", "Female"]
)

polyuria = st.selectbox("Polyuria / Sering buang air kecil", ["Yes", "No"])
polydipsia = st.selectbox("Polydipsia / Sering haus", ["Yes", "No"])
sudden_weight_loss = st.selectbox("Penurunan berat badan tiba-tiba", ["Yes", "No"])
weakness = st.selectbox("Kelemahan tubuh", ["Yes", "No"])
polyphagia = st.selectbox("Polyphagia / Sering lapar berlebihan", ["Yes", "No"])
genital_thrush = st.selectbox("Genital thrush", ["Yes", "No"])
visual_blurring = st.selectbox("Penglihatan kabur", ["Yes", "No"])
itching = st.selectbox("Gatal-gatal", ["Yes", "No"])
irritability = st.selectbox("Mudah marah / iritabilitas", ["Yes", "No"])
delayed_healing = st.selectbox("Luka sulit sembuh", ["Yes", "No"])
partial_paresis = st.selectbox("Partial paresis / kelemahan sebagian otot", ["Yes", "No"])
muscle_stiffness = st.selectbox("Kekakuan otot", ["Yes", "No"])
alopecia = st.selectbox("Alopecia / rambut rontok", ["Yes", "No"])
obesity = st.selectbox("Obesitas", ["Yes", "No"])


# =========================
# 6. Membuat Data Input
# =========================
input_data = {
    "age": age,
    "gender": gender_to_number(gender),
    "polyuria": yes_no_to_number(polyuria),
    "polydipsia": yes_no_to_number(polydipsia),
    "sudden_weight_loss": yes_no_to_number(sudden_weight_loss),
    "weakness": yes_no_to_number(weakness),
    "polyphagia": yes_no_to_number(polyphagia),
    "genital_thrush": yes_no_to_number(genital_thrush),
    "visual_blurring": yes_no_to_number(visual_blurring),
    "itching": yes_no_to_number(itching),
    "irritability": yes_no_to_number(irritability),
    "delayed_healing": yes_no_to_number(delayed_healing),
    "partial_paresis": yes_no_to_number(partial_paresis),
    "muscle_stiffness": yes_no_to_number(muscle_stiffness),
    "alopecia": yes_no_to_number(alopecia),
    "obesity": yes_no_to_number(obesity)
}

input_df = pd.DataFrame([input_data])
input_df = input_df[feature_columns]


# =========================
# 7. Prediksi
# =========================
st.divider()

if st.button("Prediksi Risiko Diabetes"):
    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0]
        prob_negative = probability[0]
        prob_positive = probability[1]
    else:
        prob_negative = None
        prob_positive = None

    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error("Hasil: Pasien Berisiko Diabetes")
    else:
        st.success("Hasil: Pasien Tidak Berisiko Diabetes")

    if prob_positive is not None:
        st.write(f"Probabilitas Tidak Berisiko: **{prob_negative * 100:.2f}%**")
        st.write(f"Probabilitas Berisiko Diabetes: **{prob_positive * 100:.2f}%**")

        st.progress(float(prob_positive))

    st.divider()

    st.subheader("Data Input yang Diproses Model")
    st.dataframe(input_df)