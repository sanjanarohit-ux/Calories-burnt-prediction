import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
import plotly.graph_objects as go

st.set_page_config(page_title="Calorie Burn Predictor", page_icon="🔥", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
h1, h2, h3, p, label, .stMarkdown, .stMetric { color: #f5f5f5 !important; }
div.stButton > button {
    background: linear-gradient(90deg, #ff512f, #f09819);
    color: white; font-weight: bold; border-radius: 12px;
    padding: 0.6em 2em; border: none; font-size: 18px; width: 100%;
}
div.stButton > button:hover { transform: scale(1.02); }
.result-box {
    background: rgba(255,255,255,0.08); border-radius: 16px;
    padding: 25px; text-align: center; margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("calories_burnt.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("calories.csv")

model = load_model()
df = load_data()

AGE_MEAN, AGE_STD = 42.7898, 16.98026416907042

st.title("🔥 Calorie Burn Predictor")
st.write("Enter your details to estimate calories burned, and explore the prediction in 3D.")

left, right = st.columns([1, 1.4])

with left:
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    age = st.slider("Age", 10, 80, 25)
    height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
    duration = st.slider("Workout Duration (min)", 1, 120, 20)
    heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 100)
    body_temp = st.slider("Body Temperature (°C)", 36.0, 42.0, 39.0, step=0.1)
    predict_clicked = st.button("🔥 Predict Calories Burned")

prediction = None
if predict_clicked:
    gender_val = 1 if gender == "Male" else 0
    user_id = random.randint(10000000, 19999999)
    age_scaled = (age - AGE_MEAN) / AGE_STD

    features = np.array([[user_id, gender_val, age, height, weight,
                           duration, heart_rate, body_temp, age_scaled]])
    prediction = model.predict(features)[0]

    with left:
        st.markdown(f"""
        <div class="result-box">
            <h2>Estimated Calories Burned</h2>
            <h1 style="color:#ff7e5f; font-size:50px;">{prediction:.2f} kcal</h1>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.subheader("3D View: Duration, Heart Rate & Calories")

    sample = df.sample(min(500, len(df)), random_state=42)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=sample["Duration"], y=sample["Heart_Rate"], z=sample["Calories"],
        mode="markers",
        marker=dict(size=4, color=sample["Calories"], colorscale="Inferno", opacity=0.6),
        name="Dataset"
    ))

    if prediction is not None:
        fig.add_trace(go.Scatter3d(
            x=[duration], y=[heart_rate], z=[prediction],
            mode="markers",
            marker=dict(size=10, color="cyan", symbol="diamond"),
            name="Your Prediction"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Duration (min)",
            yaxis_title="Heart Rate (bpm)",
            zaxis_title="Calories",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(x=0.7, y=0.95)
    )

    st.plotly_chart(fig, use_container_width=True)
