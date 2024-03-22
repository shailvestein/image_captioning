import streamlit as st 
import gdown
import numpy as np
import pickle as pkl
import os
import gc
from PIL import Image

import tensorflow as tf
import tensorflow

from tensorflow.keras.layers import Input, Embedding, RepeatVector, TimeDistributed, GRU, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model


from tensorflow import expand_dims
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

SHAPE=(600,600)
MAX_LENGTH=27
EMBEDDING_DIM=100

# downloading tokenizer from my Google drive
@st.cache(max_entries=1)
def load_tokenizer():
    tokenizer_url = "https://drive.google.com/uc?id=1F2pcPgoQOslpZ0wpPw_lMNg-3lKF6NMR"
    tokenizer_output = "tokenizer.pkl"
    gdown.download(tokenizer_url, tokenizer_output)
    with open('./tokenizer.pkl', 'rb') as f:
        tokenizer = pkl.load(f)
    return tokenizer 
    
tokenizer = load_tokenizer()
inverse_vocabulary = tokenizer.index_word
vocab_size=len(inverse_vocabulary)+1

@st.cache(max_entries=1)
def load_feature_extractor():
    print(f"Downloading feature extractor from tensorflow...", end="")
    efficientnet=EfficientNetB7(weights='imagenet')
    feature_extractor=Model(inputs=efficientnet.inputs, outputs=efficientnet.layers[-2].output)  
    print(f"done!")
    return feature_extractor
feature_extractor = load_feature_extractor()

@st.cache(max_entries=1)
def build_seq2seq_model(feature_input_shape=2560, rate=0.5, lstm_units=256, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, max_length=MAX_LENGTH):
    tf.keras.backend.clear_session()
    input_1 = Input(shape=(feature_input_shape,), name='input_1_layer')
    x1 = Dense(embedding_dim, activation='relu', name='input_1_dense_1_layer')(input_1)
    x1 = Dropout(rate, name='input_1_dropout_layer')(x1)
    x1 = RepeatVector(max_length, name='repeat_vector_layer')(x1)

    input_2 = Input(shape=(max_length,), name='input_2_layer')
    x2 = Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True, name='embedding_layer_1')(input_2)
    x2, state = GRU(lstm_units, return_state=True, return_sequences=True, name='input_2_layer_LSTM_1',)(x2)
    x2, state = GRU(lstm_units, return_state=True, return_sequences=True, name='input_2_layer_LSTM_2',)(x2, initial_state=[state])

    x2 = TimeDistributed(Dense(lstm_units, activation='softmax', name='dec_time_distributed_layer'), name='time_dsitributed_layer')(x2)
    x = Concatenate()([x1,x2])  

    x = GRU(lstm_units, return_sequences=False, name='decoder_layer_LSTM_3',)(x, initial_state=[state])
    
    x = Dense(2048, activation='relu', name='hidden_layer_1')(x)
    x = Dropout(rate, name='decoder_dropout_layer')(x)

    output = Dense(vocab_size, activation='softmax', name='dec_output_layer')(x)
    model = Model(inputs=[input_1, input_2], outputs=output, name='seq2seq_model')

    return model

# downloading trained model from my Google drive
@st.cache(max_entries=1)
def load_caption_generator():
    # downloading trained caption generator model from my google drive
    url = "https://drive.google.com/uc?id=1km3GlZ7ggKXstB0IYnCgNaC6sn3_qOtJ"
    output="image_captioner.h5"
    gdown.download(url, output, quiet=False)
    caption_generator = build_seq2seq_model()
    caption_generator.load_weights(output)
    return caption_generator
    
caption_generator = load_caption_generator()

title = """Generate Text From An Image Using Deep Learning"""

information = """Here, our AI based model will generate caption/text for the image based on the visual 
information present in uploaded image. You just need  to  upload an image it performs  magic
on images and  tells what is present  in the image without any human interference with 50% score."""

# title for the webpage
st.title(title)

# text to describe about web app
st.info(information)

st.warning(body="Disclaimer: This AI based app has its own limitations and some time result may not be enough or correct. Email your suggestion: shailvesteinsqrt@gmail.com")

# creating form to upload image 
with st.form('uploader'):
    st.text("Upload Your Color Image Here Only")
    # file uploader
    uploaded_image_file = st.file_uploader(" ", type=['jpg', 'jpeg'], accept_multiple_files=False)
    # submit button
    submitted = st.form_submit_button('Generate Text')

    
# if get images scene name button clicked
if submitted:    
    
    # appending images into list if there are more than 1 images uploaded
    # if image_file is not none
    if not uploaded_image_file is None:
        # reading image file
        image = Image.open(uploaded_image_file)
        del uploaded_image_file
        # checking dimensions of uploaded image
        if len(image.getbands()) > 1:
            with st.spinner("Generating text from image..."):

                # resizing image array
                input_image = image.resize(SHAPE, Image.Resampling.NEAREST)
                # converting image file into array
                input_image = img_to_array(input_image)
                # applying preprocessing function
                input_image = preprocess_input(input_image)
                # expanding dimension of input image 
                input_image = expand_dims(input_image, axis=0)
                # extracting features from image
                feature = np.array(feature_extractor.predict(input_image))
                del input_image
                # this will store the predicted result 
                result = "<sos>"
                for i in range(MAX_LENGTH):
                    # converting result into sequences
                    inp_seq = tokenizer.texts_to_sequences([result])[0]
                    # padding inp_sequences
                    pad_seq = pad_sequences([inp_seq], maxlen=MAX_LENGTH, padding='post')
                    # Now, predicting the captioning words for from 
                    # the image feature and padded sequences array
                    yhat = caption_generator.predict([feature, pad_seq])
                    # getting index of predicted word 
                    word_index = np.argmax(yhat, axis=-1)[0]
                    # getting word for index from inverse vocabulary
                    word = inverse_vocabulary.get(word_index)
                    # concatenating generated word to result
                    result += ' ' + word 
                    # breaking loop/prediction if <end> word detected
                    if word == '<eos>':
                        break 
			
                result = result.split(' ')
                result = ' '.join(word for word in result[1:-1])

                st.success(result)
                st.image(image)

                del result
                del feature
                gc.collect()
        else:
            st.error(body="This is a B&W image, please upload a colored image")
    else:
        st.error(body="Please upload an image before clicking on 'generate text'!!!")
else:
    pass
