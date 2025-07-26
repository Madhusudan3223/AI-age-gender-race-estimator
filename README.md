# 🧠 AI Age, Gender & Race Estimator

This project uses **Deep Learning** and **Transfer Learning** to predict a person's **Age**, **Gender**, and **Race** from an image. It leverages a pre-trained **MobileNetV2** model and is trained on the popular **UTKFace dataset**. The model is deployed as a **Streamlit web app** for real time inference.

---

## 🌐 Live Demo

👉 [Try the App](https://ai-age-gender-race-estimator-i2rarf6qkjtruwpjo2cbxj.streamlit.app/) – Upload your photo and get predictions instantly!

---

## 🧾 Features

- 🔍 Predicts **Age** (as regression)
- 🚻 Predicts **Gender** (Male/Female)
- 🌍 Predicts **Race** (White, Black, Asian, Indian, Others)
- ⚡ Real-time image inference
- 📱 MobileNetV2 for lightweight and fast prediction
- 🧠 Multi-output deep learning model

---

## 📁 Dataset

- **Source**: [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- Contains over 20,000 labeled face images
- Labels embedded in filenames: `age_gender_race_date.jpg`

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Pandas & NumPy
- Scikit-Learn
- Streamlit
- Matplotlib

---

## 🧠 Model Architecture

- **Backbone**: MobileNetV2 (frozen)
- **Outputs**:
  - `age_output` – Dense(1), loss: MAE
  - `gender_output` – Dense(1, activation='sigmoid'), loss: Binary Crossentropy
  - `race_output` – Dense(5, activation='softmax'), loss: Sparse Categorical Crossentropy

---
