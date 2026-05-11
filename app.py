import streamlit as st
import pandas as pd
import joblib
import base64

st.set_page_config(
    page_title="Student Income Predictor",
    page_icon="💰",
    layout="centered"
)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64("background.jpg")

page_bg = f"""
<style>

.stApp {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.main {{
    background-color: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 15px;
}}

h1 {{
    color: #00FFD1;
    text-align: center;
}}

p, label {{
    color: white !important;
    font-size: 16px;
}}

.stButton>button {{
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}}

.stButton>button:hover {{
    background-color: #008C9E;
}}

div[data-baseweb="select"] {{
    color: black;
}}

input {{
    color: black !important;
}}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

model = joblib.load('income_model.pkl')
scaler = joblib.load('scaler.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')

st.markdown(
    "<h1>💰 Student Monthly Income Predictor</h1>",
    unsafe_allow_html=True
)

st.info(
    "Predict student monthly income using Machine Learning 🚀"
)

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        'Age',
        15,
        40,
        20
    )

    study_hours = st.number_input(
        'Study Hours Per Day',
        0.0,
        15.0,
        5.0
    )

    cgpa = st.number_input(
        'CGPA',
        0.0,
        10.0,
        7.5
    )

    attendance = st.number_input(
        'Attendance (%)',
        0.0,
        100.0,
        85.0
    )

with col2:

    part_time_hours = st.number_input(
        'Part-Time Hours Per Week',
        0.0,
        60.0,
        10.0
    )

    internet_usage = st.number_input(
        'Internet Usage Hours',
        0.0,
        24.0,
        6.0
    )

    sleep_hours = st.number_input(
        'Sleep Hours',
        0.0,
        15.0,
        7.0
    )

    stress_level = st.selectbox(
        'Stress Level',
        [
            'Low',
            'Medium',
            'High',
            'Extreme'
        ]
    )

city = st.selectbox(
    'City',
    [
        'Bangalore',
        'Bhopal',
        'Chennai',
        'Delhi',
        'Hyderabad',
        'Indore',
        'Kolkata',
        'Mumbai',
        'Pune'
    ]
)

input_df = pd.DataFrame({

    'Age': [age],
    'Study_Hours_per_Day': [study_hours],
    'Part_Time_Hours_per_Week': [part_time_hours],
    'CGPA': [cgpa],
    'Attendance(%)': [attendance],
    'Internet_Usage_Hours': [internet_usage],
    'Sleep_Hours': [sleep_hours],
    'Stress_Level': [stress_level],
    'City': [city]

})

input_df[['Stress_Level']] = ordinal_encoder.transform(
    input_df[['Stress_Level']]
)

input_df = pd.get_dummies(
    input_df,
    columns=['City']
)

expected_columns = [

    'Age',
    'Study_Hours_per_Day',
    'Part_Time_Hours_per_Week',
    'CGPA',
    'Attendance(%)',
    'Internet_Usage_Hours',
    'Sleep_Hours',
    'Stress_Level',
    'City_Bangalore',
    'City_Bhopal',
    'City_Chennai',
    'City_Delhi',
    'City_Hyderabad',
    'City_Indore',
    'City_Kolkata',
    'City_Mumbai',
    'City_Pune'

]

for col in expected_columns:

    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_columns]

numeric_cols = [

    'Age',
    'Study_Hours_per_Day',
    'Part_Time_Hours_per_Week',
    'CGPA',
    'Attendance(%)',
    'Internet_Usage_Hours',
    'Sleep_Hours'

]

input_df[numeric_cols] = scaler.transform(
    input_df[numeric_cols]
)

if st.button('🚀 Predict Monthly Income'):

    prediction = model.predict(input_df)

    st.success(
        f'💸 Predicted Monthly Income: ₹ {prediction[0]:,.2f}'
    )
