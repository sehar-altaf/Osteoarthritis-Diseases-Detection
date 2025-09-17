import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax

# Load the model architecture and trained weights
@st.cache_resource
def load_model():
    # Base model: same as during training
    base_model = EfficientNetB3(
        weights='imagenet',            # Use ImageNet weights for base model
        include_top=False,             # Exclude top layer
        input_shape=(224, 224, 3),    # Same input size as training
        pooling=None
    )
    
    # Custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax')(x)  # 3 classes: 'Healthy', 'Minimal', 'Severe'
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile if needed (optional during inference)
    model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load the trained weights
    model.load_weights("EfficientNetB3_weights.h5")
    
    return model

# Load the model once
model = load_model()

# Class names — same as used in training after dropping classes
class_names = ['Healthy', 'Minimal', 'Severe']

# App title and description
st.title("Osteoarthritis X-ray Image Classifier")
st.write("Upload an X-ray image to classify it into one of the osteoarthritis categories.")

# Image upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    image = image.resize((224, 224))  # Resize to match training input size
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    result = class_names[predicted_class]

    # Display result
    st.write(f"### Prediction: {result}")
