import torch, pytz
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, csv
# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import streamlit as st
from PIL import Image


# Page config
st.set_page_config(page_title="ARMOR Lab Human Digital Twin Demo", layout="wide", page_icon = '/app/human_digital_twin/Deployment/EmbeddedImage.png')

app_intro = """
This website is designed for the demonstration of how functional human movement assessment can be done by using Motion Tape.
For each model/movement, the results obtained from the motion capture system serve as ground truth values to train the model.
Predictions are compared with the results obtained from the motion capture system to demonstrate the performance of Motion Tape.
* __Note__: Please change to the light theme for best user experiences. \n
"""

# Info
st.header('Introduction')
st.write(app_intro)
st.write("")
st.write("")

image = Image.open("/app/human_digital_twin/Deployment/Picture1.png")
image2 = Image.open("/app/human_digital_twin/Deployment/UCSDLogo_JSOE_Black.png")
st.sidebar.image(image2, use_column_width=True)
st.sidebar.image(image, use_column_width=True)

# Load data

with st.sidebar.expander('Models', expanded=True):
    model_name = st.selectbox(
        "Select a model",
        options=('Bicep Curl', 'Squat'))
    
    file = st.file_uploader("Upload a csv file", type="csv")

if model_name == "Bicep Curl":
    if file:
        st.write('File Received.')
    else:
        pass
        #st.header('Example:')

        #st.subheader('Motion Capture System:')
        #video_file = open('/app/human_digital_twin/Deployment/Trial09.mp4', 'rb')
        #video_bytes = video_file.read()
        #st.video(video_bytes)

        #st.subheader('Voltage Responses from Motion Tape:')
        #image3 = Image.open('/app/human_digital_twin/Deployment/Trial09.jpg')
        #st.image(image3)

        #st.subheader('Prediction:')
        #col_left, col_mid, col_right = st.columns([2, 3, 1])
        #with col_mid:
        #    image4 = Image.open('/app/human_digital_twin/Deployment/Trial9.png')
        #    st.image(image4)
elif model_name == "Squat":
    if file:
        st.write('File Received.')
    else:
        st.stop()


