import base64
import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu
def get_base64(image_path):
    with open(image_path, "rb") as image:
        return base64.b64encode(image.read()).decode()
# Load models
diabetes_model = pickle.load(open("models/diabetes_model.sav", "rb"))
diabetes_scaler = pickle.load(open("models/diabetes_scaler.sav", "rb"))

heart_model = pickle.load(open("models/heart_disease_model.sav", "rb"))
heart_scaler = pickle.load(open("models/heart_disease_scaler.sav", "rb"))

stroke_model = pickle.load(open("models/stroke_model.sav", "rb"))
stroke_scaler = pickle.load(open("models/stroke_scaler.sav", "rb"))
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction",
        ["Home","Diabetes", "Heart Disease", "Stroke"],
        icons=["home","activity", "heart", "person"],
        menu_icon="hospital",
        default_index=0
    )
if selected=="Home":
 st.title("🩺 AI-Powered Multiple Disease Prediction System")


 st.write("Welcome to your Ai Health Assistant!")
 st.write("Choose your prediction module...👈🏻:")

 st.write("🩸 Diabetes")
 st.write("❤️ Heart Disease")
 st.write("🧠 Stroke")
 st.sidebar.markdown("---")
 st.sidebar.subheader("📌 About")
 st.sidebar.info(
    """
    *Multiple Disease Prediction System*

    

    *This project leverages Machine Learning to predict the likelihood of lifestyle diseases
      ⚠️ This prediction is based on a machine learning model and is intended for educational purposes only.
      *
    """
) 
 st.markdown("---")
 st.markdown("### 🌐 Connect With Me")

 st.markdown("""
<div style="display:flex; gap:25px; font-size:20px;">
    <a href="https://instagram.com/your_username" target="_blank">📍 Instagram</a>
    <a href="https://facebook.com/your_username" target="_blank">📍 Facebook</a>
    <a href="https://twitter.com/your_username" target="_blank">📍 Twitter</a>
    <a href="https://linkedin.com/in/your_username" target="_blank">📍 LinkedIn</a>
</div>
""", unsafe_allow_html=True)

# ---------------- Diabetes Prediction ----------------

if selected == "Diabetes":

    st.header("🩸 Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0)
        glucose = st.number_input("Glucose")
        blood_pressure = st.number_input("Blood Pressure")
        skin_thickness = st.number_input("Skin Thickness")

    with col2:
        insulin = st.number_input("Insulin")
        bmi = st.number_input("BMI")
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function")
        age = st.number_input("Age", min_value=1)

    if st.button("Predict Diabetes"):

        if glucose == 0 or blood_pressure == 0 or bmi == 0:
            st.warning("⚠️ Please fill in all the required fields.")

        else:
            input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi,
                            diabetes_pedigree, age]])

            input_data = diabetes_scaler.transform(input_data)

            prediction = diabetes_model.predict(input_data)

            if prediction[0] == 1:
             st.error("🚨 Prediction: High Risk of Diabetes")

             st.warning("""
            *Recommendation*
             - 🩺 Consult a healthcare professional.
            - 🩸 Consider a blood glucose test.
            - 🥗 Follow a healthy diet.
            - 🚶 Engage in regular physical activity.
            """)

             st.caption("⚠️ This prediction is based on a machine learning model and is intended for educational purposes only.")

            else:
             st.success("✅ Prediction: Low Risk of Diabetes")

            st.info("""
            *Recommendation*
            - 🥗 Maintain a balanced diet.
            - 🚶 Exercise regularly.
            - 💧 Stay hydrated.
            - 🩺 Continue regular health check-ups.
            """)

    st.caption("⚠️ This prediction is based on a machine learning model and is intended for educational purposes only.") 
            
            # ---------------- Heart Disease Prediction ----------------

if selected == "Heart Disease":

     st.header("❤️ Heart Disease Prediction")

     col1, col2 = st.columns(2)

     with col1:
        age = st.number_input("Age", min_value=1, key="h_age")
        sex = st.selectbox("Sex", ["Female", "Male"], key="h_sex")
     sex = 0 if sex == "Female" else 1
     cp = st.selectbox("Chest Pain Type",["Typical Angina","Atypical Angina","Non-anginal pain","Asymptomatic"],key="h_cp")
     cp={
        "Typical Angina":0,
        "Atypical Angina":1,
        "Non-anginal pain":2,
        "Asymptomatic":3
    }[cp]
     trestbps = st.number_input("Resting Blood Pressure", key="h_bp")
     chol = st.number_input("Cholesterol", key="h_chol")
     fbs = st.selectbox("Fasting Blood Sugar",["No","Yes"],key="h_fbs")
     fbs= 0 if fbs=="No" else 1
     restecg = st.selectbox("Resting ECG",["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"],key="h_restecg")
     restecg={
        "Normal":0,
        "ST-T Abnormality":1,
        "Left Ventricular Hypertrophy":2
    }[restecg]

     with col2:
        thalach = st.number_input("Maximum Heart Rate", key="h_thalach")
        exang = st.selectbox("Exercise Induced Angina",["No","Yes"],key="h_exang")
        exang=0 if exang=="No" else 1
        oldpeak = st.number_input("Oldpeak", key="h_oldpeak")
        slope = st.selectbox("Slope",["Upsloping","Flat","Downsloping"],key="h_slope")
        slope={
            "Upsloping":2,
            "Flat":1,
            "Downsloping":0
        }[slope]
        caa = st.number_input("Number of Major Vessels", key="h_caa")
        thal = st.selectbox("Thal",["Normal","Fixed Defect","Reversible Defect"],key="h_thal")
        thal={
            "Normal":1,
            "Fixed Defect":2,
            "Reversible Defect":3
        }[thal]

    

     if st.button("Predict Heart Disease"):

          if age == 1 or chol == 0 or trestbps == 0 or thalach == 0:
            st.warning("⚠️ Please fill in all the required fields.")
          else:
                input_data = np.array([[age, sex, cp, trestbps, chol,
                                fbs, restecg, thalach, exang,
                                oldpeak, slope, caa, thal]])

                input_data = heart_scaler.transform(input_data)

                prediction = heart_model.predict(input_data)

                if prediction[0] == 0:
                 st.error("❤️ Prediction: High Risk of Heart Disease")

                 st.warning("""
                  *Recommendation*
                  - 🩺 Consult a cardiologist immediately.
                  - 🩸 Monitor your blood pressure regularly.
                  - 🥗 Follow a heart-healthy diet.
                  - 🚶 Engage in regular physical activity as advised by your doctor.
                  """)

                 st.caption("⚠️ This prediction is based on a machine learning model and is intended for educational purposes only.")

                else:
                 st.success("💚 Prediction: Low Risk of Heart Disease")

                 st.info("""
                  *Recommendation*
                   - 🥗 Maintain a balanced diet.
                   - 🚶 Exercise regularly.
                   - 💧 Stay hydrated.
                   - 🩺 Continue regular health check-ups.
                   """)

     st.caption("⚠️ This prediction is based on a machine learning model and is intended for educational purposes only.")
# ---------------- Stroke Prediction ----------------

if selected == "Stroke":

     st.header("🧠 Stroke Prediction")

     col1, col2 = st.columns(2)

     with col1:
          gender = st.selectbox("Gender", ["Female", "Male"], key="s_gender")
     gender = 0 if gender == "Female" else 1

     age = st.number_input("Age", min_value=1, key="s_age")
     hypertension=st.selectbox("Hypertension",["No","Yes"],key="s_hyper")
     hypertension= 0 if hypertension=="No" else 1
     heart_disease = st.selectbox("Heart Disease", ["No", "Yes"], key="s_hd")
     heart_disease = 0 if heart_disease == "No" else 1
     avg_glucose_level = st.number_input("Average Glucose Level", key="s_glucose")

     with col2:

        bmi = st.number_input("BMI", key="s_bmi")
        work_type = st.selectbox("Worktype",["Private","Self-employed","Govt-job","Children","unwmploye"],key="s_work")
        work_type={
            "Private":2,
            "Self-employed":3,
            "Govt-job":0,
            "Children":1,
            "unemploye":4
        }[work_type]
        residence_type = st.selectbox("Residence Type", ["Rural", "Urban"], key="s_residence")
     residence_type = 0 if residence_type == "Rural" else 1
     smoking_status = st.selectbox("Smoking Status",["Never Smoked","Formaly smoked","Smokes","unknown"],key="s_smoke")
     smoking_status={
    "Never Smoked":2,
    "Formaly smoked":1,
    "Smokes":3,
    "unknown":0
     }[smoking_status]
     ever_married = st.selectbox("Ever Married", ["No", "Yes"], key="s_married")
     ever_married = 0 if ever_married == "No" else 1

     if st.button("Predict Stroke"):

        if age == 1 or avg_glucose_level == 0 or bmi == 0:
            st.warning("⚠️ Please fill in all the required fields.")
        else:
            input_data = np.array([[gender,
                                    age,
                                    hypertension,
                                    heart_disease,
                                    ever_married,
                                    work_type,
                                    residence_type,
                                    avg_glucose_level,
                                    bmi,
                                    smoking_status]])

            input_data = stroke_scaler.transform(input_data)

            prediction = stroke_model.predict(input_data)

            if prediction[0] == 1:
                st.error("🧠 The person is likely to have a Stroke.")
            else:
                st.success("💚 The person is not likely to have a Stroke.")