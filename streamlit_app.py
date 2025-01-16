import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Função para carregar o modelo sem cache
def load_model(uploaded_model):
    model = tf.keras.models.load_model(uploaded_model)
    return model

# Função para processar o vídeo
def process_video(video_path, mask_path, model):
    mask = cv2.imdecode(np.frombuffer(mask_path.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    cap = cv2.VideoCapture(video_path)
    connected_components = cv2.connectedComponents(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)

    if not cap.isOpened():
        raise ValueError("Error opening video file. Check the file path.")
    
    output_path = "processed_parking_lot.mp4"
    # Preparar saída de vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Processar frames do vídeo
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        spot_crops_batch = []
        for spot in spots:
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_crops_batch.append(spot_crop)

            if len(spot_crops_batch) == BATCH_SIZE or spot == spots[-1]:
                # Pré-processar lote
                spot_crops_batch_preprocessed = [
                    cv2.resize(spot, (150, 150)) / 255.0 for spot in spot_crops_batch
                ]
                spot_crops_batch_preprocessed = np.array(spot_crops_batch_preprocessed)

                # Fazer previsões
                spot_status_batch = model.predict(spot_crops_batch_preprocessed)
                spot_status_batch = (spot_status_batch.flatten() > 0.5).astype(int)

                # Desenhar retângulos
                for i, spot in enumerate(spots[:len(spot_crops_batch)]):
                    x1, y1, w, h = spot
                    spot_status = spot_status_batch[i]
                    color = (0, 255, 0) if spot_status == EMPTY else (0, 0, 255)
                    frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

                spot_crops_batch = []

        out.write(frame)

    cap.release()
    out.release()

    return output_path

# Sidebar para upload de vídeo e máscara
st.sidebar.title("Upload de Vídeo e Máscara")
uploaded_video = st.sidebar.file_uploader("Carregar vídeo", type=["mp4", "avi", "mov"])
uploaded_mask = st.sidebar.file_uploader("Carregar imagem de máscara", type=["jpg", "png", "jpeg"])

# Carregar o modelo apenas quando o arquivo for enviado (não cacheado)
uploaded_model = st.sidebar.file_uploader("Carregar o modelo (arquivo .keras ou .h5)", type=["keras", "h5"])

# Verifique se os arquivos foram enviados e se o modelo foi carregado
if uploaded_model is not None:
    model = load_model(uploaded_model)
    st.sidebar.success("Modelo carregado com sucesso!")

    if uploaded_video is not None and uploaded_mask is not None:
        st.video(uploaded_video)

        # Processar o vídeo com a máscara
        try:
            output_path = process_video(uploaded_video, uploaded_mask, model)
            st.success("Processamento concluído! Você pode baixar o vídeo processado.")
            
            # Disponibilizar o download do vídeo processado
            with open(output_path, "rb") as file:
                st.download_button(label="Baixar vídeo processado", data=file, file_name="processed_parking_lot.mp4")
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
else:
    st.warning("Por favor, carregue o modelo, o vídeo e a máscara.")

