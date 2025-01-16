import cv2
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import streamlit as st
from tempfile import NamedTemporaryFile

# Função para obter as bounding boxes das vagas de estacionamento
def get_parking_spots_bboxes(connected_components):
    num_labels, labels = connected_components
    spots = []
    for i in range(1, num_labels):
        x1, y1, w, h = cv2.boundingRect((labels == i).astype(np.uint8))
        spots.append([x1, y1, w, h])
    return spots

# Função para criar o modelo MobileNetV2 e adaptá-lo para nossa tarefa
def create_mobilenet_model(input_shape=(150, 150, 3), num_classes=1):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)  # Dropout para evitar overfitting
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    predictions = Dense(num_classes, activation="sigmoid")(x)  # Saída binária: vazio (0) ou ocupado (1)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Congelar as camadas convolucionais do MobileNetV2 para usar apenas a parte densa
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Função para processar a imagem
def process_image(image, model, mask):
    connected_components = cv2.connectedComponents(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)
    
    spot_crops = []
    for spot in spots:
        x1, y1, w, h = spot
        spot_crop = image[y1:y1 + h, x1:x1 + w, :]
        spot_crops.append(cv2.resize(spot_crop, (150, 150)) / 255.0)
    
    spot_crops = np.array(spot_crops)
    predictions = model.predict(spot_crops)
    predictions = (predictions.flatten() > 0.5).astype(int)  # 0 (vazio) ou 1 (ocupado)

    for i, spot in enumerate(spots):
        x1, y1, w, h = spot
        color = (0, 255, 0) if predictions[i] == 0 else (0, 0, 255)  # Verde se vazio, vermelho se ocupado
        image = cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), color, 2)

    return image

# Função para processar o vídeo
def process_video(video_path, model, mask):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Erro ao abrir o vídeo. Verifique o caminho do arquivo.")
        return None

    connected_components = cv2.connectedComponents(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)
    output_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        spot_crops = []
        for spot in spots:
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_crops.append(cv2.resize(spot_crop, (150, 150)) / 255.0)

        spot_crops = np.array(spot_crops)
        predictions = model.predict(spot_crops)
        predictions = (predictions.flatten() > 0.5).astype(int)

        for i, spot in enumerate(spots):
            x1, y1, w, h = spot
            color = (0, 255, 0) if predictions[i] == 0 else (0, 0, 255)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

        output_frames.append(frame)

    cap.release()
    return output_frames

# Configuração do Streamlit
st.title("Classificação de Vagas de Estacionamento")
st.text("Carregue uma imagem ou vídeo para detectar e classificar vagas de estacionamento.")

# Criar o modelo MobileNetV2
model = create_mobilenet_model()

# Upload da máscara (área de estacionamento)
mask_path = st.file_uploader("Carregue a máscara da área de estacionamento (.png)", type=["png"])
if mask_path:
    mask = cv2.imdecode(np.frombuffer(mask_path.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # Upload de imagem
    image_file = st.file_uploader("Carregue uma imagem (.jpg, .png)", type=["jpg", "png"])
    if image_file:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image = process_image(image, model, mask)
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Imagem processada", use_column_width=True)

    # Upload de vídeo
    video_file = st.file_uploader("Carregue um vídeo (.mp4)", type=["mp4"])
    if video_file:
        temp_video = NamedTemporaryFile(delete=False)
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

        frames = process_video(temp_video_path, model, mask)
        if frames:
            st.video(temp_video_path)
            st.success("Vídeo processado com sucesso!")
        else:
            st.error("Não foi possível processar o vídeo.")







