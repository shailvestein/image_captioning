import streamlit as st 
import gdown
import numpy as np
import pickle as pkl
import os
import time
# from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Add, Concatenate
# from tensorflow.keras.models import Model
# import tensorflow as tf

from tensorflow import expand_dims
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

SHAPE=(600,600)
max_length=33

@st.cache
def load_feature_extractor():
    # downloading tensorflow pre-trained model for generating feature from image 
    efficientnet_b7=EfficientNetB7(weights='imagenet')
    feature_extractor=Model(inputs=efficientnet_b7.inputs, outputs=efficientnet_b7.layers[-2].output)
    feature_extractor.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    return feature_extractor
    
    
feature_extractor = load_feature_extractor()

@st.cache
def load_caption_generator():
    # downloading trained caption generator model from my google drive 
    url = "https://drive.google.com/uc?id=100gJejr3YcYKHQGWZd7WGSLdeprPiuEG"
    output="image_captioner.h5"
    gdown.download(url, output, quiet=False)
    time.sleep(2)
    caption_generator = "image_captioner.h5"
    return caption_generator
    
caption_generator = load_caption_generator()

@st.cache
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pkl.load(f)

    return tokenizer 
    
tokenizer = load_tokenizer()

# title for the webpage
st.title("Image Captioning Using Deep Learning")
# text to describe about web app
st.text("Here our AI based model will generate caption for the image based")
st.text("on the visual information present it the uploaded image.")
st.text("You just need to upload an image")
st.text("It performs some mathematical magic on images and tells")
st.text("what is present in the image without any huma interference.")

# creating form to upload image 
with st.form('uploader'):
     # file uploader
     image_file = st.file_uploader("Upload Your Image Here", type=['jpg'], accept_multiple_files=False)
     # submit button
     submitted = st.form_submit_button('Generate Caption')

    
# if get images scene name button clicked
if submitted:    
    # appending images into list if there are more than 1 images uploaded
    # if image_file is not none
    if not image_file is None:
        # 
        st.text('Extracting feature from image...')
        # reading image file
        image = load_img(image_file, target_size=SHAPE)
        # converting image file into array
        img = img_to_array(image)
        # applying preprocessing function
        img = preprocess_input(img)
        # expanding dimension of input image 
        img = expand_dims(img, axis=0)
        # extracting features from image
        feature = np.array(feature_extractor.predict(img))
        # 
        st.text('done')
        # 
        st.text('Generating caption....')
        # this will store the predicted result 
        result = "<start>"
        for i in range(max_length):
            # converting result into sequences
            inp_seq = tokenizer.texts_to_sequences([result])[0]
            # padding inp_sequences
            pad_seq = pad_sequences([inp_seq], maxlen=max_length, padding='post')
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
            if word == '<end>':
                break 
        st.text('done!')
        
        st.image(image, caption=result)
        
    else:
        st.text("Alert: please upload image before clicking on generate image caption!")
else:
    # if get image scene name is clicked but no images are uploaded print this messege
    st.text('Please upload an image first then click on "generate caption" button!')
