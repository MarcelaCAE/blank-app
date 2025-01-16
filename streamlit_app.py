import streamlit as st

# App Title
st.title("🎈 Parking Spot Prediction App")

# App Description
st.write("Upload a parking lot image, and the model will predict whether a spot is occupied or not.")

# Sidebar for filters and image upload
st.sidebar.title("🔧 Settings")

# Image upload in the sidebar
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Adding filters in the sidebar
brightness = st.sidebar.slider("Adjust image brightness", 0.5, 2.0, 1.0)
contrast = st.sidebar.slider("Adjust image contrast", 0.5, 2.0, 1.0)

# Check if an image has been uploaded
if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Display selected settings
    st.write("Selected Brightness:", brightness)
    st.write("Selected Contrast:", contrast)

    # Here you can apply brightness and contrast adjustments or send the image to the model.
    st.success("Image successfully uploaded! Ready to be processed by the model.")
else:
    st.warning("Please upload an image to continue.")
