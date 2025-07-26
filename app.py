import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained Keras model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("age_gender_race_model.keras")

model = load_model()

# Label encodings
age_labels = list(range(0, 116))
gender_labels = ["Male", "Female"]
race_labels = ["White", "Black", "Asian", "Indian", "Others"]

st.title("🧠 Age, Gender & Race Estimator")
st.write("Upload a clear face image to get predictions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    age = int(np.round(prediction[0][0]))
    gender = gender_labels[np.argmax(prediction[1])]
    race = race_labels[np.argmax(prediction[2])]

    st.markdown(f"### 🧓 Predicted Age: `{age}`")
    st.markdown(f"### 🚻 Predicted Gender: `{gender}`")
    st.markdown(f"### 🌍 Predicted Race: `{race}`")
