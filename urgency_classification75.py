#Import library
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

#Load dataset
data = pd.read_csv('Dataset Pengaduan Masyarakat.csv')
data = data[['complaint', 'urgency']]

#Get data
x = data['complaint']
y = [ 0 if i=='Not Urgent' else 1 for i in data['urgency'] ]

#Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

vocab_size = 1000
embedding_dim = 64
max_length = 50
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

#Tokenization
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

train_seq = tokenizer.texts_to_sequences(x_train)
test_seq = tokenizer.texts_to_sequences(x_test)

#Padding
train_pad = pad_sequences(train_seq, padding=padding_type, truncating=trunc_type, maxlen=max_length)
test_pad = pad_sequences(test_seq, padding=padding_type, truncating=trunc_type, maxlen=max_length)

train_pad = np.array(train_pad)
train_label = np.array(y_train)
test_pad = np.array(test_pad)
test_label = np.array(y_test)

#Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.LSTM(70),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

model.fit(train_pad, train_label, epochs=50, batch_size=32, validation_data=(test_pad, test_label), callbacks=[es])

eval_result = model.evaluate(test_pad, test_label)
print("Test Loss:", eval_result[0])
print("Test Accuracy:", eval_result[1])

#Save model h5
model.save('save_model75.h5')

#Save tokenizer
with open('tokenization75.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
