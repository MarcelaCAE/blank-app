import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tempfile
import time

# Constants for parking spot status
EMPTY = 0
NOT_EMPTY = 1

# Load the pre-trained MobileNet model (assuming the model is available locally)
model_path ='C:/Users/meite/Ironhack Course/templates/modelo_parking_lot (1) (4).keras'

MODEL = tf.keras.models.load_model(model_path)

# Function to preprocess image for model prediction
def preprocess_image_for_model(image):
    # Resize the image to match the input size of MobileNet
    img_resized = cv2.resize(image, (150, 150)) / 255.0
    return np.expand_dims(img_resized, axis=0)

# Function to detect parking spots and apply model predictions
def detect_parking_spots(frame, mask):
    # Find connected components in the mask image
    connected_components = cv2.connectedComponents(mask, 4, cv2.CV_32S)
    num_labels, labels = connected_components

    # Extract bounding boxes for each connected component (parking spot)
    spots = []
    for i in range(1, num_labels):
        x1, y1, w, h = cv2.boundingRect((labels == i).astype(np.uint8))
        spots.append([x1, y1, w, h])
    
    # Crop each parking spot and apply the model to check if it's empty or not
    for spot in spots:
        x1, y1, w, h = spot
        spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
        
        # Preprocess the spot crop and make prediction
        spot_crop_resized = preprocess_image_for_model(spot_crop)
        spot_status = MODEL.predict(spot_crop_resized)[0][0]

        # Draw bounding box and label (green for empty, red for not empty)
        if spot_status == EMPTY:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  # Green for empty
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)  # Red for not empty

    return frame

# App title
st.title("Parking Lot Detection")

# Sidebar for file upload with unique key for selectbox
st.sidebar.title("Settings")
file_type = st.sidebar.selectbox("Choose file type", ["Image", "Video"], key="file_type_selectbox")

uploaded_file = None

if file_type == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="image_uploader")
    
elif file_type == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="video_uploader")

# Create an expander for raw data and loading message
with st.expander("Raw Data"):
    # If a file is uploaded
    if uploaded_file is not None:
        # Simulate loading time with a progress bar
        with st.spinner('Processing your file...'):
            time.sleep(2)  # Simulate the processing time, remove or adjust as needed
            
            if file_type == "Image":
                # Load the uploaded image
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # Assume mask is predefined or generate from uploaded image
                mask = cv2.imread('path_to_mask_image/mask.png', 0)  # Update path to the mask image

                # Detect parking spots in the image and classify as empty or occupied
                result_frame = detect_parking_spots(img_array, mask)
                
                # Display the processed image
                st.image(result_frame, caption="Processed Image with Parking Spot Detection", use_container_width=True)
            
            elif file_type == "Video":
                # Process the video (you can use a similar approach for video as above)
                st.write("Video processing is not yet supported in this snippet.")
                # Optionally, you can adapt the video processing loop to detect parking spots frame by frame.







