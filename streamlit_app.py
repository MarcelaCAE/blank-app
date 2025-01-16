import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from google.colab.patches import cv2_imshow

# Constants
EMPTY = 0
NOT_EMPTY = 1
MODEL_PATH = '/content/drive/MyDrive/computer_vision/parking_lot_contents/parking/saved_models/modelo_parking_lot.keras'
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Function to get parking spot bounding boxes
def get_parking_spots_bboxes(connected_components):
    num_labels, labels = connected_components
    slots = []
    for i in range(1, num_labels):
        x1, y1, w, h = cv2.boundingRect((labels == i).astype(np.uint8))
        slots.append([x1, y1, w, h])
    return slots

# Function to predict parking spot status (empty or not)
def empty_or_not(spot_bgr):
    img_resized = cv2.resize(spot_bgr, (150, 150))
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = img_resized / 255.0
    y_output = MODEL.predict(img_resized)
    return EMPTY if y_output[0] == 0 else NOT_EMPTY

# Sidebar for uploading files (image or video)
st.sidebar.title('Parking Lot Detection')
uploaded_file = st.sidebar.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    # Check if the file is an image or video
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        # Process image
        img = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        # Display image
        st.image(img, caption="Uploaded Image", channels="BGR", use_column_width=True)

        # Detect parking spots (dummy process)
        mask = np.zeros_like(img)  # Placeholder for actual mask
        connected_components = cv2.connectedComponents(mask, 4, cv2.CV_32S)
        spots = get_parking_spots_bboxes(connected_components)

        for spot in spots:
            x1, y1, w, h = spot
            spot_crop = img[y1:y1 + h, x1:x1 + w]
            status = empty_or_not(spot_crop)
            color = (0, 255, 0) if status == EMPTY else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, 2)

        st.image(img, caption="Parking Spot Detection Result", channels="BGR", use_column_width=True)

    elif uploaded_file.type == "video/mp4":
        # Process video
        video_bytes = uploaded_file.read()
        st.video(video_bytes, format="video/mp4", start_time=0)

        # Open video stream
        cap = cv2.VideoCapture(uploaded_file.name)

        # Placeholder for actual mask and connected components
        mask = np.zeros_like((640, 480), dtype=np.uint8)  # Replace with actual mask
        connected_components = cv2.connectedComponents(mask, 4, cv2.CV_32S)
        spots = get_parking_spots_bboxes(connected_components)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect parking spot status and draw boxes on the frame
            for spot in spots:
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w]
                status = empty_or_not(spot_crop)
                color = (0, 255, 0) if status == EMPTY else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

            # Display the frame
            st.image(frame, channels="BGR", caption="Parking Lot Video", use_column_width=True)
        
        cap.release()

else:
    st.info("Please upload an image or a video to get started.")


