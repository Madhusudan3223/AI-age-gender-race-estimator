import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import tempfile

# Load model
@st.cache_resource
def load_age_gender_race_model():
    return load_model("age_gender_race_model.keras")

model = load_age_gender_race_model()

# Face detection using Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_and_preprocess(image):
    img_cv = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    face = img_cv[y:y+h, x:x+w]
    face = cv2.resize(face, (128, 128))
    face = face / 255.0
    return np.expand_dims(face, axis=0)

# Streamlit UI
st.set_page_config(page_title="AI Age, Gender & Race Estimator", layout="centered")
st.markdown("## 🧠 AI Age, Gender & Race Estimator")
st.write("Upload a face image and let the AI predict age, gender, and race.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    face_input = detect_face_and_preprocess(image)

    if face_input is None:
        st.error("❌ No face detected. Please upload a clear face image.")
    else:
        age_pred, gender_pred, race_pred = model.predict(face_input)
        age = int(age_pred[0][0])
        gender = "Male" if gender_pred[0][0] < 0.5 else "Female"
        race_labels = ["White", "Black", "Asian", "Indian", "Others"]
        race = race_labels[np.argmax(race_pred)]

        st.subheader("📊 Prediction Results")
        st.write(f"**Estimated Age:** {age}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Race:** {race}")
