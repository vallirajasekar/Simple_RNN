import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model 

from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}


model=load_model('simple_rnn_imdb.h5')


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review
    
    
## Creating the Prediction Function 


def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction


st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a Movie Review to classify it as positive or Negative')


user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    prediction=model.predict(prediction)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    
    # Display the Result 
    
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction Score:{prediction[0][0]}')
else:
    st.write('Please Enter the Movie Review')