import streamlit as st
import tensorflow as tf
import numpy as np
from predict_text import *

from PIL import Image

from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten

INPUT_SHAPE = (224,224,3)
TARGET_SHAPE = (224,224)
@st.cache()
def load_feature_extractor():
  eNetB7 = EfficientNetB7(include_top=False, 
                          weights='imagenet', 
                          input_shape=(INPUT_SHAPE))

  for layer in eNetB7.layers:
      layer.trainable = False


  # inputs = Input(shape=INPUT_SHAPE, name='input layer')

  inputs = eNetB7.inputs
  x = eNetB7.layers[-2].output
  x = Dense(4096, activation='relu', name='dense_1')(x)
  outputs = GlobalAveragePooling2D(name='global_pooling_layer')(x)



  feature_extractor = Model(inputs=inputs, outputs=outputs)
  return feature_extractor
  
  
feature_extractor = load_feature_extractor()
st.title('Text generator from image')
st.text('Please upload your image to generate text based on the visual context present in the image')
st.warning('Warning: This project is under development phase')
with st.form('uploader'):
  image = st.file_uploader('upload image', type=['jpg', 'png', 'jpeg'], accept_multiple_files=False)
  submitted = st.form_submit_button('generate features')

def extract_feature(image):
  image = img_to_array(image)
  image = preprocess_input(image)
  image = tf.expand_dims(image, axis=0)
  extracted_feature = feature_extractor.predict(image)
  return extracted_feature[0]
  
if submitted:
  if image:
    image = Image.open(image) 
    image = image.resize(TARGET_SHAPE)
    feat = extract_feature(image)
    st.image(image)
    st.text(feat)
  else:
    st.text('Please upload an image')
  
else:
  st.text('')
