# Image Captioning Using Deep Learning

This project is developed now and using sequence-to-sequence model for prediction in the backend.

Webapp link deployed on streamlit: https://shailvestein-image-captioning-streamlit-app-o0osw7.streamlitapp.com/

Data set downloaded from kaggle link https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset

## 1. Data loading and pre-preocessing:
* We loaded captions into the memory and performed some text pre-processsing on them.
* And saved them into the disk for further use during training.
* After pre-processing performed EDA.
* Made some conclusions like: max number = 27, of words in a sentence to be consider because 99.9% of captions having words <= 27 in both train and val, found out some words that are present frequently in the captions using "Word Cloud".

## 2. Feature Extraction:
* Initialised efficientnet b7 pre-trained model for feature extraction.
* and then downloaded pre-trained efficientnet b7 for feature extraction for train, val and test set images.
* Extracted features and saved them into the disk for further use during model training.

## 3. Training:
* First of all, we loaded and pre-processed images captions and extracted features from disk.
* Defined custom dataloader / generator and sequence-to-sequence model for image captioning using tensorflow v2.5
* During training the pre-processed and extracted features will be feed to model as input.
* After training model, define a caption generator function which will generate text from the image.
* After this, the BLEU score is calculated on the actual and predicted captions using NLTK.
* By seeing, BLEU score for train and val dataset and made conclusion that this model is generalized model because the difference between train and val BLEU score is nearly 1%.
* Generated text on 50 test images and found model is performing good.
