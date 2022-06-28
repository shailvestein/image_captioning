import wget
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Add, Concatenate, Embedding, Input
from tensorflow.keras.models import Model

@st.cache()
def load_model():
  # encoder block
  # image feature block
  inputs1 = Input(shape=(1024,), name='inputs1_layer')
  enc1 = Dropout(0.5, name='dropout_1')(inputs1)
  enc1 = Dense(1024, activation='relu', name='input1_dense')(enc1)

  max_length = 49
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

  ## defin model
  model = Model(inputs=[inputs1, inputs2], outputs=outputs, name='seq2seq_model')
  
  return model.load_weights('sample_1.h5')
  
  
  

text_predictor_url = "https://drive.google.com/file/d/1dI1jBVo0Bj1GzHNo7UV-Bc2d4XJhVVit/view?usp=sharing"

text_predictor = wget.download(text_predictor_url)
if text_predictor:
  print('Model loaded")
else:
  print('model not loaded')
