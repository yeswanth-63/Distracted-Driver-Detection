import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model from the .h5 file
model = tf.keras.models.load_model('distracted_driver.h5')

# Define the class labels
class_labels = ['safe driving', 'texting - right', 'talking on the phone - right', 'texting - left',
                'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind',
                'hair and makeup', 'talking to passenger']

# Define the Streamlit app layout
st.title('Distracted Driver Detection')
st.write('Upload an image of a driver to predict the distraction category')

# Create a file upload button
uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize the image to match the input size of the model
    image = np.array(image)  # Convert the PIL image to a NumPy array
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    return class_labels[predicted_class], confidence

# Make predictions if an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    predicted_class, confidence = predict(image)
    st.write(f'Predicted Class: {predicted_class}')
    st.write(f'Confidence: {confidence}')
