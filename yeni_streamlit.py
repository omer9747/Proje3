import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="https://cdn-icons-png.freepik.com/512/5935/5935638.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "mailto:omerisik9747@gmail.com",
        "About": "This project allows people to learn the diagnosis of heart disease in advance by entering certain parameters. However, it should not be forgotten that this model does not replace a medical authority, so please do not worry. Have a healthy day :)"
    }
)

st.title("HeartDisease Predictor")
st.markdown("If you want to measure your risk of heart disease as an individual, you are at the right place. Because the **accuracy of the algorithm we use is over 86%.**")
st.image("https://www.hopkinsmedicine.org/-/media/images/health/3_-wellness/heart-health/heart-hero.jpg?h=500&iar=0&mh=500&mw=1300&w=1297&hash=F41798F1F4CB19003E3A592502480D34")

st.subheader("Algorithms That We Used")
st.markdown("1. Logistic Regression 2. K-Neighbors 3. Perceptron 4. SVC 5. RandomForest 6. DecisionTree 7. XGboost")
st.subheader("How We Use Them?")
st.markdown("The variables we looked at when choosing the most suitable algorithm here were std score and cross-validation score, and we decided that the most suitable algorithm for our model was the RandomForest algorithm.")
st.image("https://media.geeksforgeeks.org/wp-content/uploads/20240130162938/random.webp")

st.header("Data Dictionary")
st.markdown("**HeartDisease (Output class) => [1: Heart disease, 0: Normal]**")
st.markdown("**Age => Patient's age [years]**")
st.markdown("**Sex => Gender of the patient [M: Male, F: Female]**")
st.markdown("**ChestPainType => Chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]**")
st.markdown("**RestingBP => Resting Blood Pressure**")
st.markdown("**FastingBS => Fasting Blood Sugar**")
st.markdown("**RestingECG => Resting electrocardiogram results [Normal: Normal, ST: ST-T wave abnormality (T wave inversions and/or ST elevation or depression > 0.05 mV), LVH: Possible or definite left ventricular hypertrophy according to Estes criteria]**")
st.markdown("**Max HR => Maximum heart rate reached [numerical value between 60 and 202]**")
st.markdown("**ExerciseAngina => Exercise-induced angina [Y: Yes, N: No]**")
st.markdown("**Oldpeak => ST [Numerical value measured as depression]**")
st.markdown("**ST_Slope => Hill exercise ST segment slope [Up: Rising, Flat: Flat, Down: Descending]**")

# Load data
df = pd.read_pickle("df.pkl")
st.table(df.sample(5))

# Sidebar inputs
st.sidebar.markdown("Enter your values below")
Name = st.sidebar.text_input("Name")
Surname = st.sidebar.text_input("Surname")
Age = st.sidebar.number_input("Age", min_value=0, max_value=99)
Sex = st.sidebar.selectbox("What is your gender", ("Male", "Female"))
ChestPainType = st.sidebar.selectbox("what is your Chest Pain Type",("TA","ATA","NAP","ASY"))
RestingBP = st.sidebar.text_input("Blood Pressure Level", help="Enter your resting blood pressure level")
Cholesterol = st.sidebar.number_input("Cholesterol Level")
FastingBS = st.sidebar.text_input("Blood Sugar Level", help="You can go to the nearest pharmacy and have it measured.")
RestingECG = st.sidebar.selectbox("What is your Resting Electrocardiogram",("Normal","ST","LVH"))
MaxHR = st.sidebar.number_input("Max HR", min_value=60, max_value=202)
ExerciseAngina = st.sidebar.selectbox("Do you have Exercise Angina?",("Yes","No"))
Oldpeak = st.sidebar.text_input("Oldpeak", help="ST [Numerical value measured as depression]")
ST_Slope = st.sidebar.selectbox("What is your ST_Slope?",("Up","Flat","Down"))

# Load model
rf_model = load('rf_model.pkl')

# Prepare input data
input_df = pd.DataFrame({
    'Age': [Age],
    'Gender': [Sex],
    'ChestPainType':[ChestPainType],
    'RestingBP': [RestingBP],
    'Cholesterol':[Cholesterol],
    'FastingBS': [FastingBS],
    'RestingECG': [RestingECG],
    'MaxHR': [MaxHR],
    'ExerciseAngina': [ExerciseAngina],
    'Oldpeak': [Oldpeak],
    'ST_SLope': [ST_Slope]
})

df = df[['Age', 'Gender', 'ChestPainType','RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_SLope']]
result_df = pd.concat([df, input_df])


######
result_df = pd.get_dummies(result_df, columns=['Sex','ChestPainType','ExerciseAngina','RestingECG','ST_Slope'], drop_first=True, dtype=int)


# Standardize data
std_scale = StandardScaler()
scaled_result_df = std_scale.fit_transform(result_df)




# Prediction
test_df = scaled_result_df[-1].reshape(1, 16)
test_pred = rf_model.predict(test_df)
test_pred_proba = rf_model.predict_proba(test_df)

st.header("Output")

if st.sidebar.button("Submit"):
    st.info("Your results are written below")

    from datetime import date, datetime
    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    results_df = pd.DataFrame({
        'Name': [Name],
        'Surname': [Surname],
        'Date': [today],
        'Time': [time],
        'Age': [Age],
        'Gender': [Sex],
        'ChestPainType':[ChestPainType],
        'RestingBP': [RestingBP],
        'Cholesterol':[Cholesterol],
        'FastingBS': [FastingBS],
        'RestingECG': [RestingECG],
        'MaxHR': [MaxHR],
        'ExerciseAngina': [ExerciseAngina],
        'Oldpeak': [Oldpeak],
        'ST_SLope': [ST_Slope],
        'Prediction': test_pred,
        'Normal': test_pred_proba[:, :1],
        'HeartDisease': test_pred_proba[:,1:]
    })
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: "NORMAL" if x == 0 else "You may have a heart disease, please have a check-up")

    st.table(results_df)


else:
    st.markdown("Please click the *Submit Button*!")
