import streamlit as st
from tensorflow.keras.models import load_model 
import cv2
import numpy as np
import os

model_path = "leaf_disease_model.h5"
model = load_model(model_path)

dataset_path = r"C:\Users\User\OneDrive\Desktop\Leaf_disease detection model\dataset\Train"
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

st.title("🌿 Leaf Disease Detection System")

st.write("Upload a leaf image to detect disease and get solution & precautions.")

disease_info = {

    "Black spot": {
        "Solution": "Apply copper fungicide and prune diseased leaves.",
        "Precaution": "Ensure proper air circulation and avoid overhead watering."
    },

    "Citrus leaf disease": {
        "Solution": "Use copper-based fungicides and ensure proper soil drainage.",
        "Precaution": "Avoid water logging and monitor regular irrigation."
    },

    "Leaf rust": {
        "Solution": "Apply triazole fungicides and remove infected leaves.",
        "Precaution": "Avoid overhead irrigation and provide good spacing."
    },

    "Powdery": {
        "Solution": "Spray sulfur-based fungicides and ensure airflow.",
        "Precaution": "Avoid wet leaves and prune nearby branches."
    },

    "Mildew": {
        "Solution": "Use systemic fungicides and reduce moisture.",
        "Precaution": "Avoid overcrowding and water only at soil level."
    },

    "Anthracnose": {
        "Solution": "Apply copper fungicide; remove infected parts.",
        "Precaution": "Avoid overhead watering and disinfect pruners."
    },

    "Leaf gall": {
        "Solution": "Remove and destroy infected leaves.",
        "Precaution": "Improve airflow and avoid too much nitrogen fertilizer."
    },

    "Cassava mosaic": {
        "Solution": "Use virus-free planting material and destroy affected plants.",
        "Precaution": "Control whiteflies and avoid mixing infected crops."
    }
}
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)


    predictions = model.predict(img_resized)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions[0]) * 100

    disease_data = disease_info.get(predicted_class, {
        "Solution": "No solution found for this disease.",
        "Precaution": "No precautionary advice available."
    })

    st.success(f"### 🏷 Predicted Class: {predicted_class}")
    st.write(f"### 🎯 Confidence: {confidence:.2f}%")

    st.write(f"**🛠 Suggested Solution:** {disease_data['Solution']}")
    st.write(f"**🛡 Precaution:** {disease_data['Precaution']}")

    st.bar_chart(predictions[0])
