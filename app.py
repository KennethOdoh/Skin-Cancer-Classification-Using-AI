import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import preprocessing
from keras.models import load_model
from keras.activations import sigmoid
import os
import h5py

st.title(':blue[MLFlow Prediction App]')
st.header('Skin Cancer Prediction')
st.text("Upload a skin cancer Image for image classification")

def main():
    file_uploaded = st.file_uploader('Choose the file', type = ['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    # classifier_model = mod('MlFlow.h5')
    classifier_model = tf.keras.models.load_model('MlFlow_softmax_sparse.h5')
    shape = ((180, 180, 3))
    # model = tf.keras.Sequential(classifier_model)
    test_image = image.resize((180, 180))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    # test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    class_names = ['actinic keratosis',
                   'basal cell carcinoma',
                   'dermatofibroma',
                   'melanoma',
                   'nevus',
                   'pigmented benign keratosis',
                   'seborrheic keratosis',
                   'squamous cell carcinoma',
                   'vascular lesion']

    predictions = classifier_model.predict(test_image)
    predictions = tf.where(predictions < 0.5, 0, 1)
    # scores = tf.nn.softmax(predictions)
    scores =predictions.numpy()
    image_class = class_names[np.argmax(scores)]
    result = 'The image predicted is : {}'.format(image_class)
    return result

if __name__ =="__main__":
    main()

