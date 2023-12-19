#Import library
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Load ML model
model = tf.keras.models.load_model("save_model75.h5")

#Load tokenizer
with open('tokenization75.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = FastAPI()

class InputData(BaseModel):
    complaint: str

@app.get("/")
def test_deploy():
    return {"PredictAPI finally deployed"}

@app.post("/predict")
def predict(data: InputData):

    sentence = ([data.complaint])

    #Remove repeated words
    def repeated_words(text):
        word = text.split()
        uniq_words = list(set(word))
        return ' '.join(uniq_words)

    new_sentence = [repeated_words(text) for text in sentence]

    #Sequenced and padding input
    sequence = tokenizer.texts_to_sequences(new_sentence)
    pad_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')

    #Predict input
    predict_value = model.predict(pad_sequence)
    percentage = round(predict_value[0][0] * 100, 2)
    urgency = "Urgent" if predict_value[0][0] > 0.5 else "Not Urgent"

    return {"percentage (%)": percentage, "urgency": urgency}
