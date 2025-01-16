import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to process the image
def process_image(image):
    # Convert the image to a format that OpenCV can process
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Simulate parking lot detection (just a simple example)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    return thresholded

# Function to process the video
def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simulate parking lot detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Show the processed frame
        st.image(thresholded, channels="BGR")
        
        # Pause to show the video frame by frame
        if st.button("Pause"):
            break
    
    cap.release()

# App title
st.title("Parking Lot Detection")

# Sidebar for file upload
st.sidebar.title("Settings")
file_type = st.sidebar.selectbox("Choose file type", ["Image", "Video"])

uploaded_file = None

if file_type == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
elif file_type == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# If a file is uploaded
if uploaded_file is not None:
    if file_type == "Image":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        processed_image = process_image(image)
        st.image(processed_image, caption="Processed Image", use_column_width=True)
        
    elif file_type == "Video":
        st.video(uploaded_file)
        process_video(uploaded_file)

