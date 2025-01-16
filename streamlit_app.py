import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

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
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(video_file.read())
        tmpfile_path = tmpfile.name
    
    cap = cv2.VideoCapture(tmpfile_path)
    
    raw_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simulate parking lot detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        raw_data.append(thresholded)
    
    cap.release()
    return raw_data

# App title
st.title("Parking Lot Detection")

# Create an expander below the title
with st.expander("Loading Data... Please wait while the file is processed."):
    
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
        # Simulate loading time with a progress bar
        with st.spinner('Processing your file...'):
            time.sleep(2)  # Simulate the processing time, remove or adjust as needed

            if file_type == "Image":
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                processed_image = process_image(image)
                st.image(processed_image, caption="Processed Image", use_column_width=True)
                
                # Display raw data in an expandable section
                with st.expander("Raw Data (Image)"):
                    st.write("Raw pixel data of the processed image:")
                    st.write(processed_image)
            
            elif file_type == "Video":
                st.video(uploaded_file)
                raw_data = process_video(uploaded_file)
                
                # Display raw data in an expandable section
                with st.expander("Raw Data (Video)"):
                    st.write("Raw pixel data of the processed video:")
                    for i, frame in enumerate(raw_data):
                        st.write(f"Frame {i}:")
                        st.image(frame, channels="BGR")
                
        # Hide the loading message after processing
        st.write("Data processed successfully!")



