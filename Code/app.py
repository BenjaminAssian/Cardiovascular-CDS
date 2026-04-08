import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Cardiovascular Risk Decision Support System",
    page_icon="❤️",
    layout="wide"
)

# -------------------------
# LOAD MODEL AND DATA
# -------------------------
model = joblib.load("../model/heart_model.pkl")
df = pd.read_csv("../data/heart.csv")

# -------------------------
# HEADER
# -------------------------
st.title("❤️ Cardiovascular Risk Decision Support System")

st.markdown("""
This clinical decision support dashboard uses machine learning to estimate a patient's
risk of developing cardiovascular disease using physiological health indicators.

⚠️ This tool is intended to assist clinicians and does **not replace medical diagnosis**.
""")

# -------------------------
# DASHBOARD METRICS
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Model Accuracy", "87.5%")
col2.metric("Dataset Size", len(df))
col3.metric("Features Used", df.shape[1] - 1)

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3 = st.tabs(["Dashboard", "Data Explorer", "Model Insights"])

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 20, 100, 50)
resting_bp = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
max_hr = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 5.0, 1.0)

fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar >120 mg/dL", [0,1])

sex = st.sidebar.selectbox("Sex", ["M","F"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA","NAP","ASY","TA"])
exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Y","N"])
st_slope = st.sidebar.selectbox("ST Slope", ["Up","Flat","Down"])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal","ST","LVH"])

# -------------------------
# FORMAT INPUT DATA
# -------------------------
input_data = pd.DataFrame({
    'Age':[age],
    'RestingBP':[resting_bp],
    'Cholesterol':[cholesterol],
    'FastingBS':[fasting_bs],
    'MaxHR':[max_hr],
    'Oldpeak':[oldpeak],
    'Sex_M':[1 if sex=="M" else 0],
    'ChestPainType_ATA':[1 if chest_pain=="ATA" else 0],
    'ChestPainType_NAP':[1 if chest_pain=="NAP" else 0],
    'ChestPainType_TA':[1 if chest_pain=="TA" else 0],
    'RestingECG_Normal':[1 if resting_ecg=="Normal" else 0],
    'RestingECG_ST':[1 if resting_ecg=="ST" else 0],
    'ExerciseAngina_Y':[1 if exercise_angina=="Y" else 0],
    'ST_Slope_Flat':[1 if st_slope=="Flat" else 0],
    'ST_Slope_Up':[1 if st_slope=="Up" else 0]
})

# -------------------------
# TAB 1: DASHBOARD
# -------------------------
with tab1:

    st.header("Patient Risk Prediction")

    if st.sidebar.button("Predict Heart Disease Risk"):

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        st.write(f"Predicted Risk Probability: **{probability:.2%}**")

# -------------------------
# TAB 2: DATA EXPLORER
# -------------------------
with tab2:

    st.header("Dataset Exploration")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    fig1, ax1 = plt.subplots()
    ax1.hist(df["Age"])
    ax1.set_title("Age Distribution")
    col1.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(df["Cholesterol"])
    ax2.set_title("Cholesterol Distribution")
    col2.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    df["HeartDisease"].value_counts().plot(kind="bar", ax=ax3)
    ax3.set_title("Heart Disease Cases")
    st.pyplot(fig3)

# -------------------------
# TAB 3: MODEL INSIGHTS
# -------------------------
with tab3:

    st.header("Model Insights")

    df_encoded = pd.get_dummies(df, drop_first=True)

    st.subheader("Feature Correlation Heatmap")

    fig4, ax4 = plt.subplots(figsize=(10,8))
    sns.heatmap(df_encoded.corr(), cmap="coolwarm", ax=ax4)

    st.pyplot(fig4)