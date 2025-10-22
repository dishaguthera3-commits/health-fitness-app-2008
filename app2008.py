import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="AI Health & Diet Recommender",
    layout="wide",
    page_icon="💪"
)

# ---------------- Load pre-trained models ----------------
pkl_path = os.path.join(os.getcwd(), "sample_models.pkl")
try:
    with open(pkl_path, "rb") as f:
        models = pickle.load(f)
except FileNotFoundError:
    st.error("❌ sample_models.pkl not found! Please upload it in the same folder as app.py")
    st.stop()

reg_calories = models['reg_calories']
reg_protein = models['reg_protein']
reg_fat = models['reg_fat']
reg_carb = models['reg_carb']
clf_cardio = models['clf_cardio']
clf_strength = models['clf_strength']
clf_mobility = models['clf_mobility']
X_cols = models['X_cols']

# ---------------- Sidebar: User Inputs ----------------
st.sidebar.header("📝 Enter your details")
weight = st.sidebar.number_input("Weight (kg)", 40.0, 150.0, 70.0)
height = st.sidebar.number_input("Height (cm)", 140.0, 210.0, 170.0)
age = st.sidebar.number_input("Age", 18, 60, 25)
gender = st.sidebar.selectbox("Gender", ["male","female"])
exercise_level = st.sidebar.selectbox("Exercise level", ["sedentary","light","moderate","active","very active"])
water_cups = st.sidebar.number_input("Water cups/day", 0, 15, 6)

# ---------------- Prepare Input ----------------
input_dict = {
    'weight': weight,
    'height': height,
    'age': age,
    'water_cups': water_cups,
    'gender_female': 1 if gender=='female' else 0,
    'gender_male': 1 if gender=='male' else 0,
    'exercise_level_sedentary': 1 if exercise_level=='sedentary' else 0,
    'exercise_level_light': 1 if exercise_level=='light' else 0,
    'exercise_level_moderate': 1 if exercise_level=='moderate' else 0,
    'exercise_level_active': 1 if exercise_level=='active' else 0,
    'exercise_level_very active': 1 if exercise_level=='very active' else 0
}

input_df = pd.DataFrame([input_dict])
for col in X_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X_cols]

# ---------------- Predict ----------------
calories = reg_calories.predict(input_df)[0]
protein = reg_protein.predict(input_df)[0]
fat = reg_fat.predict(input_df)[0]
carb = reg_carb.predict(input_df)[0]

cardio = clf_cardio.predict(input_df)[0]
strength = clf_strength.predict(input_df)[0]
mobility = clf_mobility.predict(input_df)[0]

# ---------------- Display Header ----------------
st.title("💪 AI Health & Diet Recommender")
st.markdown(
    "Get personalized diet and exercise recommendations based on your body data. "
    "This app is aligned with **SDG 3 — Good Health & Wellbeing**."
)

# ---------------- Nutrition Cards ----------------
st.subheader("🥗 Daily Nutrition Targets")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Calories (kcal)", round(calories))
col2.metric("Protein (g)", round(protein))
col3.metric("Fat (g)", round(fat))
col4.metric("Carbs (g)", round(carb))

# ---------------- Exercise Cards ----------------
st.subheader("🏋️ Personalized Exercise Plan")
col1, col2, col3 = st.columns(3)
col1.info(f"💓 Cardio:\n{cardio}")
col2.success(f"🏋️ Strength:\n{strength}")
col3.warning(f"🧘 Mobility:\n{mobility}")

# ---------------- Hydration ----------------
st.subheader("💧 Hydration Recommendation")
recommended_water_ml = weight * 30
recommended_cups = round(recommended_water_ml / 250)
st.write(f"You reported: {water_cups} cups (~{water_cups*250} ml)")
st.write(f"Recommended: ~{recommended_cups} cups (~{recommended_water_ml:.0f} ml) per day")
if water_cups < recommended_cups:
    st.warning("Try drinking more water to meet daily needs 💧")
else:
    st.success("Your water intake meets or exceeds the recommendation ✅")

# ---------------- BMI Section ----------------
bmi = weight / ((height/100)**2)
st.subheader("📊 BMI & Category")
st.metric("Your BMI", f"{bmi:.1f}")

# BMI category with colors
if bmi < 18.5:
    st.info("Category: Underweight")
elif bmi < 25:
    st.success("Category: Normal weight")
elif bmi < 30:
    st.warning("Category: Overweight")
else:
    st.error("Category: Obesity")

# ---------------- Progress bars for BMI visualization ----------------
st.subheader("BMI Progress Bar")
bmi_percentage = min(max((bmi-10)/30, 0), 1)
st.progress(bmi_percentage)

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "This app uses AI to provide personalized recommendations. "
    "It is aligned with Sustainable Development Goal 3 — Good Health & Wellbeing."
)
