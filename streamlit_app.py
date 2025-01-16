import streamlit as st

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

# Carregar o modelo salvo
model_path = '/content/drive/MyDrive/computer_vision/parking_lot_contents/parking/saved_models/modelo_parking_lot.keras'
MODEL = tf.keras.models.load_model(model_path)

# Função para processar a imagem e realizar a previsão
def predict_parking_spot_status(img):
    # Redimensiona a imagem para o formato que o modelo espera
    img_resized = cv2.resize(img, (150, 150))
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = img_resized / 255.0  # Normaliza a imagem

    # Realiza a previsão
    prediction = MODEL.predict(img_resized)

    # Retorna 0 (vazio) ou 1 (ocupado) com base na previsão
    return "Ocupado" if prediction[0] > 0.5 else "Vazio"

# Título do aplicativo
st.title("🎈 Aplicativo de Previsão de Vagas de Estacionamento")

# Descrição do aplicativo
st.write("Carregue uma imagem do estacionamento e o modelo irá prever se a vaga está ocupada ou não.")

# Carregar a imagem usando o widget do Streamlit
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Carregar a imagem usando o OpenCV
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)

    # Exibir a imagem carregada
    st.image(img, caption='Imagem Carregada', use_column_width=True)

    # Prever o status da vaga
    status = predict_parking_spot_status(img_array)

    # Mostrar o resultado
    st.write(f"O status da vaga é: **{status}**")

