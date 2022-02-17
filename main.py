from PIL import Image
import numpy as np
import streamlit as st
from keras.models import load_model
import tensorflow as tf


def SSIMLoss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ssimloss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return ssimloss


# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    image = image.reshape(1, 128, 128, 1)
    image = image / 255

    return image[0]


# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=["jpg", "png"])
col1, col2 = st.columns(2)

# Checking the Format of the page
if uploadFile is not None:
    #   st.write(type(uploadFile))

    # Perform your Manupilations (In my Case applying Filters)
    img_input = load_image(uploadFile)
    col1.header("Input Image")
    col1.image(img_input)
    cnn_model = load_model("models/preprocessed_data_cnn_model.h5")
    autoencoder = load_model("models/v0-1.h5")
    pred = autoencoder.predict(img_input)

    col2.header("reconstructed image")
    pred = np.array(pred).reshape(128, 128, 1)
    prediction = cnn_model.predict(np.array(img_input).reshape(1, 128, 128, 1))
    col2.image(pred)
    COL1, COL2, COL3, COL4 = st.columns(4)
    loss = SSIMLoss(img_input, pred)
    label = "SSIM Loss value: {:.3f}"
    label_classes = "Predicted Class: {}"
    classes = np.argmax(prediction, axis=1)

    COL2.write(label.format(loss))
    COL2.write(label_classes.format(classes[0] + 1))

    CNN_COL1, CNN_COL2 = st.columns(2)
    img = Image.open(f"processed_data/templates/{classes[0]+1}.png")
    CNN_COL1.header("input image")
    CNN_COL1.image(img_input)
    CNN_COL2.header("predicted_image")
    CNN_COL2.image(img)
    img = np.array(img).reshape(128, 128, 1)
    img = img / 255

    loss2 = SSIMLoss(img_input, img)
    label = "SSIM Loss value: {:.3f}"
    label_classes = "Predicted Class: {}"
    COL1, COL2, COL3, COL4 = st.columns(4)
    COL2.write(label.format(loss2))
    label = "Ratio loss values: {:.2f}"

    COL2.write(label.format(loss / loss2))
    result = ""
    if loss / loss2 < 0.65:
        result = "anomaly"
    else:
        result = "normal"

    COL2.write(result)


else:
    st.write("Make sure you image is in JPG/PNG Format.")
