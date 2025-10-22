import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Health & Diet Recommender", layout="wide")

# ---------------- Load pre-trained models ----------------
with open("sample_models.pkl", "rb") as f:
    models = pickle.load(f)

reg_calories = models['reg_calories']
reg_protein = models['reg_protein']
reg_fat = models['reg_fat']
reg_carb = models['reg_carb']
clf_cardio = models['clf_cardio']
clf_strength = models['clf_strength']
clf_mobility = models['clf_mobility']
X_cols = models['X_cols']

# ---------------- Sidebar: User inputs ----------------
st.sidebar.header("Enter your details")
weight = st.sidebar.number_input("Weight (kg)", 40.0, 150.0, 70.0)
height = st.sidebar.number_input("Height (cm)", 140.0, 210.0, 170.0)
age = st.sidebar.number_input("Age", 18, 60, 25)
gender = st.sidebar.selectbox("Gender", ["male","female"])
exercise_level = st.sidebar.selectbox("Exercise level", ["sedentary","light","moderate","active","very active"])
water_cups = st.sidebar.number_input("Water cups/day", 0, 15, 6)

# ---------------- Prepare input dataframe ----------------
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

# ---------------- Display Results ----------------
st.title("💪 AI Health & Diet Recommender")
st.markdown("Personalized diet and exercise recommendations based on your body data.")

st.subheader("🥗 Daily Nutrition Targets")
st.write(f"**Calories:** {round(calories)} kcal")
st.write(f"**Protein:** {round(protein)} g | **Fat:** {round(fat)} g | **Carbs:** {round(carb)} g")

st.subheader("🏋️ Personalized Exercise Plan")
st.write(f"**Cardio:** {cardio}")
st.write(f"**Strength training:** {strength}")
st.write(f"**Mobility & Stretching:** {mobility}")

st.subheader("💧 Hydration Recommendation")
recommended_water_ml = weight * 30  # ~30 ml per kg
recommended_cups = round(recommended_water_ml / 250)
st.write(f"You reported: {water_cups} cups (~{water_cups*250} ml)")
st.write(f"Recommended: ~{recommended_cups} cups (~{recommended_water_ml:.0f} ml) per day")

# ---------------- Optional: BMI visualization ----------------
bmi = weight / ((height/100)**2)
st.subheader("📊 BMI")
st.write(f"Your BMI: {bmi:.1f}")
fig, ax = plt.subplots(figsize=(6,1.2))
ax.set_xlim(10,40)
ax.set_ylim(0,1)
ax.axis('off')
ranges = [(10,18.5,'Under'), (18.5,25,'Normal'), (25,30,'Over'), (30,40,'Obese')]
colors = ['#ffd1dc','#c8f7c5','#fff2b2','#ffb3b3']
for (start,end,_),c in zip(ranges,colors):
    ax.fill_betweenx([0,1],[start],[end], color=c)
ax.plot([bmi,bmi],[0,1], color='black')
st.pyplot(fig)