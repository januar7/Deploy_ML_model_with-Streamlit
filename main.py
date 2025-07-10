# main.py (Example - you'll have your own implementation)
import torch
import streamlit as st
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# Definisiin dulu arsitektur model MU!!
class Model_NN_simple(nn.Module):
    def __init__(self, num_classes=10): # Updated to default to 10 for CIFAR-10
        super(Model_NN_simple, self).__init__()
        # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)
        # So, input features after flattening will be 32 * 32 * 3 = 3072
        self.fc1 = nn.Linear(32 * 32 * 3, 128) # First hidden layer
        self.fc2 = nn.Linear(128, 64)          # Second hidden layer
        self.fc3 = nn.Linear(64, num_classes)  # Output layer (raw logits for 10 classes)
        # No softmax here, as nn.CrossEntropyLoss handles it internally

    def forward(self, X):
        # Flatten the input image from (batch_size, channels, height, width)
        # to (batch_size, channels * height * width)
        X = X.view(X.size(0), -1)

        # Pass through fully connected layers with ReLU activation
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X) # Return raw logits
        return X

def load_trained_model(model_path, num_classes):
    model = Model_NN_simple(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}. Please ensure '{model_path}' exists and is compatible.")
        st.stop() # Stop the app if model fails to load
    return model

def preprocess_image_for_model(image: Image.Image):
    # CIFAR-10 specific transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # CIFAR-10 images are 32x32
        transforms.ToTensor(),
        # CIFAR-10 normalization values
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]) 
    ])
    # Add a batch dimension (B, C, H, W)
    return transform(image).unsqueeze(0)

def predict_image(model, input_tensor):
    with torch.no_grad(): # Disable gradient calculation for inference
        output = model(input_tensor)
    return output