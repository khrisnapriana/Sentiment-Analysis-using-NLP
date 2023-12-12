from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np

app = FastAPI()

class InputData(BaseModel):
    complaint: str

# Load the model
model = tf.keras.models.load_model("new_model77.h5")

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=500, oov_token='<OOV>')

@app.post("/predict")
def predict(data: InputData):

    tokenizer.fit_on_texts([data.complaint])

    # Tokenize and pad the input data
    sequence = tokenizer.texts_to_sequences([data.complaint])
    pad_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')

    # Make a prediction
    predict_value = model.predict(pad_sequence)
    percentage = round(predict_value[0][0] * 100, 2)

    # Convert the prediction to urgency label
    urgency = "Urgent" if predict_value[0][0] > 0.5 else "Not Urgent"

    return {"percentage (%)": percentage, "urgency": urgency}