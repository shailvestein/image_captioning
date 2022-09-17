import streamlit as st 
import gdown
import numpy as np
import pickle as pkl
import os
from PIL import Image

from tensorflow.keras.layers import Input, Embedding, RepeatVector, TimeDistributed, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

from tensorflow import expand_dims
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

SHAPE=(600,600)
MAX_LENGTH=33
EMBEDDING_DIM=200


@st.cache
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pkl.load(f)
    return tokenizer 
    
tokenizer = load_tokenizer()
inverse_vocabulary = tokenizer.index_word
vocab_size=len(inverse_vocabulary)+1

@st.cache
def load_feature_extractor():
    # downloading tensorflow pre-trained model for generating feature from image 
    # url = 'https://drive.google.com/uc?id=1-050q5AQWBArHiDZf6yRH6bXk29KoWPV'
    # output = 'feature_extractor.h5'
    efficientnet_b7=EfficientNetB7(weights='imagenet')
    feature_extractor=Model(inputs=efficientnet_b7.inputs, outputs=efficientnet_b7.layers[-2].output)
    optmizer = tf.keras.optimizers.Adam(1e-5)
    # feature_extractor.compile(optimizer=optmizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return feature_extractor
    
    
feature_extractor = load_feature_extractor()

@st.cache
def build_model(feature_input_shape, vocab_size, units, max_length, embedding_dim):
    input1 = Input(shape=feature_input_shape, name='feature_input_layer')
    x1=Dense(units, activation='relu', name='dense_layer_1_after_feature_input')(input1)
    x1=RepeatVector(max_length, name='repeat_vector_layer')(x1)
    input2 = Input(shape=(max_length,), name='caption_input_layer')
    x2=Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True, trainable=False, name='Embedding_layer')(input2)
    x2=LSTM(units, return_sequences=True, name='lstm_layer_1')(x2)
    x2=TimeDistributed(Dense(embedding_dim, name='densly_time_distributed_layer'), name='time_distributed_layer')(x2)

    x=Concatenate(name='concate_layer')([x1,x2])
    x=LSTM(units, return_sequences=True)(x)
    x=LSTM(units)(x)
    x=Dense(units, activation='relu', name='dense_layer_1_after_concat')(x)
    output=Dense(vocab_size, activation='softmax', name='output_layer')(x)
    model=Model(inputs=[input1, input2], outputs=output, name='image_captioning_model')
    return model

@st.cache
def load_caption_generator():
    # downloading trained caption generator model from my google drive 
    url = "https://drive.google.com/uc?id=10AkZ2UTReklr_lDlJ1R8jUWszr_OCscG"
    output="image_captioner.h5"
    gdown.download(url, output, quiet=False)
    caption_generator = build_model(feature_input_shape=2560, vocab_size=vocab_size, units=256, max_length=MAX_LENGTH, embedding_dim=EMBEDDING_DIM)
    caption_generator.load_weights("image_captioner.h5")
    return caption_generator
    
caption_generator = load_caption_generator()



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
     uploaded_image_file = st.file_uploader("Upload Your Image Here", type=['jpg'], accept_multiple_files=False)
     # submit button
     submitted = st.form_submit_button('Generate Caption')

    
# if get images scene name button clicked
if submitted:    
    # appending images into list if there are more than 1 images uploaded
    # if image_file is not none
    if not uploaded_image_file is None:
        # 
        st.text('Extracting feature from image...')
        # reading image file
        image = Image.open(uploaded_image_file)
        # resizing image array
        resized_image = image.resize(SHAPE, Image.Resampling.NEAREST)
        # converting image file into array
        image_array = img_to_array(resized_image)
        # applying preprocessing function
        preprocessed_image = preprocess_input(image_array)
        # expanding dimension of input image 
        expanded_dim_image = expand_dims(preprocessed_image, axis=0)
        # extracting features from image
        feature = np.array(feature_extractor.predict(expanded_dim_image))
        # 
        st.text('done')
        # 
        st.text('Generating caption....')
        # this will store the predicted result 
        result = "<start>"
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
            if word == '<end>':
                break 
                
                
        st.text('done!')
        st.text('')
        
        result = result.split(' ')
        output = ' '.join(word for word in result[1:-1])
        st.text(f'Caption: {output}')
        st.image(image)
        
    else:
        st.text(" ")
else:
    # if get image scene name is clicked but no images are uploaded print this messege
    st.text('Alert: please upload image before clicking on generate image caption!')
