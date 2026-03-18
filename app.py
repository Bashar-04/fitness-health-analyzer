import streamlit as st
from streamlit_lottie import st_lottie
import requests
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("model.pkl")  # نفس اللي خزنت
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="🌟 Fitness AI App",
    page_icon="💪",
    layout="wide"
)

# -----------------------------
# Load Lottie Animation (background)
# -----------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_bg = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
st_lottie(lottie_bg, height=350, key="bg")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
    <div style='text-align: center; color: white;'>
        <h1>Fitness & Health Analyzer</h1>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
st.header("Enter Your Data")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 70, 25)
    weight = st.slider("Weight (kg)", 40, 150, 70)
    height = st.slider("Height (cm)", 140, 210, 170)

with col2:
    sleep = st.slider("Sleep (hours)", 0, 12, 7)
    exercise = st.slider("Exercise (hours/week)", 0, 14, 3)
    steps = st.slider("Daily Steps", 0, 20000, 6000)
    calories = st.slider("Calories", 1000, 4000, 2000)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Analyze"):
    with st.spinner("Analyzing your health..."):
        time.sleep(2)

    # Prepare Data
    user_data = np.array([[age, weight, height, sleep, exercise, steps, calories]])
    user_scaled = scaler.transform(user_data)
    cluster = model.predict(user_scaled)[0]

    # Correct mapping based on your data analysis
    cluster_names = {
        0: ("⚖️ Average Lifestyle", "warning"),
        1: ("🛑 Sedentary", "error"),
        2: ("💪 Active & Healthy", "success")
    }

    msg, level = cluster_names[cluster]
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.error(msg)

    # Radar Chart
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[sleep, exercise, steps/2000, calories/1000],
        theta=['Sleep','Exercise','Steps','Calories'],
        fill='toself',
        name="Your Profile"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(12,14,10,4)])),
        showlegend=False,
        title="Your Health Profile"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

