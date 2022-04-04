import streamlit as st
from streamlit_player import st_player
import train
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pathlib

model = train.train()
uploaded_file = st.file_uploader("Choose files", accept_multiple_files=False)
img = tf.keras.utils.load_img(
    uploaded_file, target_size=(180, 180)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

st.write("ST Player:")
left, right = st.columns(2)
with left:
    link = st.text_input('Enter a Youtube Link: ')
    if (not link):
        link = "https://youtu.be/CmSKVW1v0xM"
    st_player(str(link))
with right:
    st.text_input('Enter a Soundclod Link: ')
    st_player("https://soundcloud.com/imaginedragons/demons")

# for uploaded_file in uploaded_files:
#     xyz = uploaded_file.getvalue().decode("utf-8")
    # render_mol(xyz)


