import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# --- CONFIGURACIÓN DE LA CNN ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Salida: 32 x 14 x 14
        x = self.pool(F.relu(self.conv2(x))) # Salida: 64 x 7 x 7
        x = x.view(-1, 64 * 7 * 7)            # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- FUNCIONES DE UTILIDAD ---
def predict(image, model):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    return prediction

# --- INTERFAZ DE STREAMLIT ---
st.title("🔢 Clasificador de Dígitos MNIST con CNN")
st.write("Sube una imagen de un dígito escrito a mano (0-9) para que la IA lo identifique.")

# Cargar modelo (Simulado: en un entorno real cargarías un .pth pre-entrenado)
@st.cache_resource
def load_model():
    model = CNN()
    # Aquí deberías cargar los pesos: model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    
    if st.button('Clasificar'):
        label = predict(image, model)
        st.success(f"La predicción es: {label}")
