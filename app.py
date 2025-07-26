import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model with compile=False to avoid custom object errors
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("age_gender_race_model.keras", compile=False)
    return model

model = load_model()

# Define label mappings
gender_labels = ["Male", "Female"]
race_labels = ["White", "Black", "Asian", "Indian", "Others"]

# Image preprocessing
def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("🧠 AI Age, Gender & Race Estimator")
st.markdown("Upload a face image and let the AI predict age, gender, and race.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        input_data = preprocess_image(image)
        age_pred, gender_pred, race_pred = model.predict(input_data)

        predicted_age = int(age_pred[0][0])
        predicted_gender = gender_labels[np.argmax(gender_pred)]
        predicted_race = race_labels[np.argmax(race_pred)]

    st.subheader("🧾 Prediction Results")
    st.write(f"**Estimated Age:** {predicted_age}")
    st.write(f"**Gender:** {predicted_gender}")
    st.write(f"**Race:** {predicted_race}")
