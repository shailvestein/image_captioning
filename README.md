# image_captioning

This project is under development phase.

Prototype webapp link deployed on streamlit: https://shailvestein-image-captioning-streamlit-app-o0osw7.streamlitapp.com/

This Text generation works as follows:
1. Takes image as an input and extracts features.
2. Extracted features are then passed to the sequenced model which predicts the next word and input looks like: [extracted_features_array, "\<start>"]
3. Now, our model predicts the next word after <start> and then the next input will be [extracted_features_array, "\<start> word1"] and this process is going as long as the the model predicts "\<end>" word.
4. Now, we have generated a sentence whick looks like: "<start> word1 word2 word3 ... \<end>"
