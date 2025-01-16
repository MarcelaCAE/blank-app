import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# Função para detectar vagas de estacionamento
def get_parking_spots_bboxes(connected_components):
    num_labels, labels = connected_components

    slots = []
    for i in range(1, num_labels):
        x1, y1, w, h = cv2.boundingRect((labels == i).astype(np.uint8))
        slots.append([x1, y1, w, h])

    return slots

# Função para prever a ocupação das vagas
def process_frame(frame, spots, model, batch_size=64):
    spot_crops_batch = []
    for spot in spots:
        x1, y1, w, h = spot

        # Recortar a vaga do frame
        spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
        spot_crops_batch.append(spot_crop)

        # Processar em lotes
        if len(spot_crops_batch) == batch_size or spot == spots[-1]:
            spot_crops_batch_preprocessed = [
                cv2.resize(spot, (150, 150)) / 255.0 for spot in spot_crops_batch
            ]
            spot_crops_batch_preprocessed = np.array(spot_crops_batch_preprocessed)

            # Previsões para o lote
            spot_status_batch = model.predict(spot_crops_batch_preprocessed)
            spot_status_batch = (spot_status_batch.flatten() > 0.5).astype(int)

            for i, spot in enumerate(spots[:len(spot_crops_batch)]):
                x1, y1, w, h = spot
                spot_status = spot_status_batch[i]
                if spot_status == 0:  # Vaga vazia
                    frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  # Verde
                else:  # Vaga ocupada
                    frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)  # Vermelho

            spot_crops_batch = []
    return frame

# Função principal para processar vídeo
def process_video(video_path, mask_path, model, output_path='output_video.avi'):
    mask = cv2.imread(mask_path, 0)
    connected_components = cv2.connectedComponents(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo. Verifique o caminho do arquivo.")
        return

    # Configuração do vídeo de saída
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, spots, model)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Vídeo processado e salvo em: {output_path}")

# Função para processar uma única imagem
def process_image(image_path, mask_path, model):
    mask = cv2.imread(mask_path, 0)
    connected_components = cv2.connectedComponents(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)

    frame = cv2.imread(image_path)
    frame = process_frame(frame, spots, model)

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Caminhos para o modelo, máscara e entrada do usuário
model = create_mobilenet_model()  # Use o modelo criado anteriormente
mask_path = r'C:\Users\meite\Downloads\parking-lot (4)\parking\mask_crop.png'

# Escolha da entrada pelo usuário
input_type = input("Digite 'imagem' para processar uma imagem ou 'video' para processar um vídeo: ").strip().lower()
if input_type == 'imagem':
    image_path = input("Digite o caminho para a imagem: ").strip()
    process_image(image_path, mask_path, model)
elif input_type == 'video':
    video_path = input("Digite o caminho para o vídeo: ").strip()
    process_video(video_path, mask_path, model)
else:
    print("Opção inválida. Digite 'imagem' ou 'video'.")








