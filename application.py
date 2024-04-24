import pickle
import streamlit as st

# Load scaler and model from pickle files (replace with your file paths)
scaler = pickle.load(open("standardScalar.pkl", "rb"))
model = pickle.load(open("modelForPrediction.pkl", "rb"))

# Title and header
st.title("Diabetes Prediction App")
st.header("Enter your information:")

# User input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Blood Glucose", min_value=0)
bloodpressure = st.number_input("Diastolic Blood Pressure", min_value=0)
skinthickness = st.number_input("Skin Thickness (mm)", min_value=0)
insulin = st.number_input("Insulin (μU/mL)", min_value=0)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age (years)", min_value=0)

# Button to trigger prediction
predict_button = st.button("Predict")

if predict_button:
  # Prepare data for prediction
  new_data = scaler.transform([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])

  # Make prediction
  predict = model.predict(new_data)

  # Display result
  if predict[0] == 1:
    result = "Diabetic"
  else:
    result = "Non-Diabetic"

  st.write(f"**Prediction:** {result}")

st.write("**Note:** This is a basic prediction tool and should not be used for medical diagnosis.")
