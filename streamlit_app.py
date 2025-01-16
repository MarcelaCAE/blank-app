import streamlit as st
import tensorflow as tf
import requests
from io import BytesIO

import streamlit as st
import gdown
import tensorflow as tf
from io import BytesIO

# Link do Google Drive (certifique-se de pegar o link de compartilhamento público)
file_url = "https://github.com/MarcelaCAE/blank-app/raw/main/modelo_parking_lot%20(1).keras"

def download_model_from_drive(url):
    try:
        # Baixa o arquivo diretamente
        gdown.download(url, 'modelo_parking_lot.keras', quiet=False)
        # Carregar o modelo Keras após o download
        model = tf.keras.models.load_model('modelo_parking_lot.keras')
        return model
    except Exception as e:
        st.error(f"Erro ao baixar ou carregar o modelo: {e}")
        return None

# Baixar e carregar o modelo
model = download_model_from_drive(file_url)

if model:
    st.write("Modelo carregado com sucesso!")

