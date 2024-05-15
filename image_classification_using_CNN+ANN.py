import streamlit as st
import keras
import numpy as np
import pandas as pd
import sklearn
import cv2
import os
import tempfile
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
uploaded_file=None

model=load_model(r'C:\Users\LENOVO\OpenCV\best_weights_cnn1.keras')

uploaded_file=st.file_uploader("Upload your image",type=["jpg", "jpeg", "png"])



labels=["lifeboat","ladybug","pizza","bell pepper","school bus","koala","espresso","panda","orange","sports car"]
le.fit(labels)
if st.button("Upload"):

        if uploaded_file is not None:
                
                # Read the image using OpenCV
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)

                # Display the image
                st.image(img, channels="BGR", caption="Uploaded Image")


                image=le.inverse_transform([np.argmax(model.predict(img[np.newaxis]))])[0]
                st.write("Predicted Image :" ,image.upper())
        else:
                st.write("❌Please upload an image")