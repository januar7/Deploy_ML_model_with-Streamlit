# app.py
import os
import streamlit as st
import torch
from PIL import Image
import numpy as np # Might be needed for some output processing

# Import functions and classes from your main.py file
# Ensure main.py contains Model_CNN_simple, load_trained_model,
# preprocess_image_for_model, and predict_image,
# and that they are compatible with CIFAR-10.
from main import Model_NN_simple, load_trained_model, preprocess_image_for_model, predict_image
# impor fungsi dan kelas dari file python main.py

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="CIFAR-10 NN Image Predictor",
    page_icon="🖼️",
    layout="centered"
)

# --- Global Constants ---
# PASTIKAN SELALU GUNAKAN BOBOT MODEL YANG SUDAH DILATIH!!
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR,'model_NN_simple.pth')
#PATH LOcal
# MODEL_PATH = 'D:\KULIAH\Data Science\Belajar_Deploy_ML\model_NN_simple.pth' # Pastikan ini adalah path bobot model yang telah disimpan
NUM_CLASSES = 10 # Jumlah kelas, Contoh:Terdapat 10 kelas

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --- Load the model (cached to run only once) ---
# st.cache_resource is perfect for large objects like models that don't change
@st.cache_resource
def get_model():
    # Make sure load_trained_model is configured to load a model for 10 classes
    model = load_trained_model(MODEL_PATH, NUM_CLASSES)
    return model

model = get_model() # Load the model when the app starts

# --- Streamlit UI ---
st.title("🖼️ CIFAR-10 Image Prediction App")
st.markdown("Upload an RGB image (preferably 32x32) to get a prediction from the trained CNeN model.")
st.markdown("This model is trained on the CIFAR-10 dataset, which includes categories like airplanes, cars, birds, etc.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB') # Ensure the image is RGB
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    # st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")

    # Add a spinner while predicting
    with st.spinner('Predicting...'):
        # Preprocess the image using the function from main.py
        # This function should resize to 32x32 and apply CIFAR-10 normalization
        input_tensor = preprocess_image_for_model(image)

        # Make prediction using the function from main.py
        output = predict_image(model, input_tensor)

    # Process and display the output for multi-class classification (CIFAR-10)
    st.subheader("Prediction Results:")

    
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    predicted_probability = probabilities[predicted_class_idx].item() * 100

    st.success(f"Predicted Class: **{predicted_class_name}**")
    st.info(f"Confidence: {predicted_probability:.2f}%")

    st.markdown("---")
    st.subheader("All Class Probabilities:")
    # Display all probabilities in a table
    # Data disimpan dalam dictionary dengan key-value pairs
    prob_data = {
        "Class Name": CLASS_NAMES,
        "Probability (%)": [f"{p.item()*100:.2f}" for p in probabilities]
    }
    # Using st.dataframe for better presentation, though st.table works too.
    # If you get an error here, uncomment `import pandas as pd` and replace st.table
    # import pandas as pd
    # st.dataframe(pd.DataFrame(prob_data))
    st.table(prob_data)


else:
    st.info("Oii, upload gambarnya dulu!!")

st.markdown("---")
st.markdown("This app uses a simple Neural Network for image prediction.")
st.markdown("Developed using Streamlit and PyTorch.")