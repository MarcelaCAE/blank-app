import streamlit as st
import tensorflow as tf
import requests
from io import BytesIO

# URL do modelo no GitHub (link raw)
model_url = "https://github.com/MarcelaCAE/blank-app/raw/main/modelo_parking_lot%20(1).keras"

# Função para baixar o modelo
def load_model_from_github(url):
    # Baixar o arquivo
    response = requests.get(url)
    if response.status_code == 200:
        # Abrir o modelo a partir do conteúdo baixado
        model = tf.keras.models.load_model(BytesIO(response.content))
        return model
    else:
        st.error("Erro ao baixar o modelo.")
        return None

# Baixar o modelo
model = load_model_from_github(model_url)

if model:
    st.write("Modelo carregado com sucesso!")

    # Aqui você pode adicionar o código para usar o modelo para fazer previsões, etc.
    # Exemplo de como fazer uma previsão (supondo que o modelo seja para prever algo)
    # Você pode colocar inputs de dados, fazer previsões, etc.



