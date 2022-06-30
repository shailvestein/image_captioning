import streamlit as st
import tensorflow as tf
import numpy as np
import wget
from PIL import Image

from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten

INPUT_SHAPE = (224,224,3)
TARGET_SHAPE = (224,224)
IMAGE_IMP_SHAPE = 4096

@st.cache()
def load_feature_extractor():
  eNetB7 = EfficientNetB7(include_top=False, 
                          weights='imagenet', 
                          input_shape=(INPUT_SHAPE))

  for layer in eNetB7.layers:
      layer.trainable = False
      
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
  
@st.cache()
def load_text_predictor():

  # encoder block
  inputs1 = Input(shape=IMAGE_IMP_SHAPE, name='inputs1_layer')
  enc1 = Dropout(0.5, name='dropout_1')(inputs1)
  enc1 = Dense(1024, activation='relu', name='input1_dense')(enc1)
  max_length = max_length
  
  # sequence2sequence block
  inputs2 = Input(shape=(max_length,), name='inputs2_layer')
  enc2 = Embedding(input_dim=vocab_size, output_dim=256, input_length=max_length, mask_zero=True, name='embedding_layer')(inputs2)
  enc2 = LSTM(units=256, return_sequences=True, name='LSTM_1')(enc2)
  enc2 = LSTM(256, return_sequences=True, name='LSTM_2')(enc2)
  enc2 = LSTM(256, return_sequences=True, name='LSTM_3')(enc2)
  enc2 = LSTM(units=256, name='LSTM_4')(enc2)
  enc2 = Dropout(0.5, name='dropout_2')(enc2)

  concat = Concatenate()([enc1, enc2])

  # decoder block
  dec = Dense(512, activation='relu', name='dec_dense1')(concat)
  dec = Dropout(0.5, name='dropout_3')(dec)
  outputs = Dense(vocab_size, activation='softmax', name='dec_output')(dec)

  ## initializing model
  text_predictor_model = Model(inputs=[inputs1, inputs2], outputs=outputs, name='seq2seq_model')
  
  text_predictor_url = "https://drive.google.com/file/d/1cIfXdgSWjlseXkU24w5BKp-scXsQbCtm/view?usp=sharing"
  text_predictor = wget.download(text_predictor_url)
  text_predictor_model.load_model(text_predictor)
  return text_predictor_model

text_predictor_model = load_text_predictor()

@st.cache()
def predict_caption(img_features, text_predictor_model):
    text = 'sos'
    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        pad_seq = pad_sequences([seq], maxlen=max_length, padding='post')
        yhat = text_predictor.predict([img_features, pad_seq])
        y_pred = np.argmax(yhat)
        word = inverse_vocab.get(y_pred)
        text += ' ' + word
        if word == 'eos':
            break
    return text

if submitted:
  if image:
    image = Image.open(image) 
    image = image.resize(TARGET_SHAPE)
    feat = extract_feature(image)
    predicted_text = predict_caption(feat, text_predictor_model)
    st.image(image)
    st.text(feat)
    st.text(predicted_text)
  else:
    st.text('Please upload an image')
  if text_predictor_model:
    st.text('text predictor loaded')
    # st.text(text_predictor.predict(image))
    
  else:
    st.text('text predictor not loaded')
  
else:
  st.text('')
