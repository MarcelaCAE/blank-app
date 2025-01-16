import os
import cv2
import numpy as np
import tensorflow as tf
from google.colab import files
from google.colab.patches import cv2_imshow

# Constants
EMPTY = 0
NOT_EMPTY = 1
BATCH_SIZE = 64

# Load Model
MODEL_PATH = '/content/drive/MyDrive/computer_vision/parking_lot_contents/parking/saved_models/modelo_parking_lot.keras'
MODEL = tf.keras.models.load_model(MODEL_PATH)

def get_parking_spots_bboxes(connected_components):
    """Get bounding boxes for parking spots from connected components."""
    num_labels, labels = connected_components
    slots = []
    for i in range(1, num_labels):
        x1, y1, w, h = cv2.boundingRect((labels == i).astype(np.uint8))
        slots.append([x1, y1, w, h])
    return slots

def process_video(video_path, mask_path, output_path):
    """Process the video to identify empty and occupied parking spots."""
    mask = cv2.imread(mask_path, 0)
    cap = cv2.VideoCapture(video_path)
    connected_components = cv2.connectedComponents(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)

    if not cap.isOpened():
        raise ValueError("Error opening video file. Check the file path.")

    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process video frames
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
                # Preprocess batch
                spot_crops_batch_preprocessed = [
                    cv2.resize(spot, (150, 150)) / 255.0 for spot in spot_crops_batch
                ]
                spot_crops_batch_preprocessed = np.array(spot_crops_batch_preprocessed)

                # Predict batch
                spot_status_batch = MODEL.predict(spot_crops_batch_preprocessed)
                spot_status_batch = (spot_status_batch.flatten() > 0.5).astype(int)

                # Draw rectangles
                for i, spot in enumerate(spots[:len(spot_crops_batch)]):
                    x1, y1, w, h = spot
                    spot_status = spot_status_batch[i]
                    color = (0, 255, 0) if spot_status == EMPTY else (0, 0, 255)
                    frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

                spot_crops_batch = []

        out.write(frame)

    cap.release()
    out.release()

# Sidebar Interface
print("Upload the mask image for the parking lot:")
mask_file = files.upload()
mask_path = list(mask_file.keys())[0]

print("Upload the video for processing:")
video_file = files.upload()
video_path = list(video_file.keys())[0]

output_path = "processed_parking_lot.mp4"

try:
    process_video(video_path, mask_path, output_path)
    print("Processing complete! Download your processed video below.")
    files.download(output_path)
except Exception as e:
    print(f"An error occurred: {e}")
