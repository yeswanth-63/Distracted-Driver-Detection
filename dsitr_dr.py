import streamlit as st
import tensorflow as tf
from PIL import Image
import pickle
model = pickle.load(open('driver_distraction_model.pkl',"rb"))
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
def preprocess(image):
    # Resize the image to the input shape expected by your model
    image = image.resize((100, 100))

    # Convert the image to an array of pixel values
    image_array = kimage.img_to_array(image)

    # Normalize the pixel values (assuming your model expects values in the range [0, 1])
    image_array = image_array / 255.0

    # Expand the dimensions of the image to match the input shape expected by your model
    image_array = tf.expand_dims(image_array, 0)

    return image_array


def predict(image):
    # Preprocess the image (e.g., resize, normalize, etc.)
    processed_image = preprocess(image)

    # Make predictions using the loaded model
    predictions = model.predict(processed_image)

    return predictions

def main():
    st.title("Distracted Driver Detection")
    st.write("Upload an image of a driver and get the predicted class.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform prediction
        predictions = predict(image)

        # Display the predicted class and probabilities
        class_names = ['C0', 'C1', 'C2','C3', 'C4', 'C5','C6', 'C7', 'C8','C9']  # Replace with your actual class names
        st.write("Predicted class:", class_names[predictions.argmax()])
        st.write("Class probabilities:", predictions)

if __name__ == '__main__':
      main()
