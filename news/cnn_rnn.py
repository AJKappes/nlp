import pandas as pd
import numpy as np
import glob
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
tokenizer = Tokenizer(oov_token = '<00V>')

d_path = glob.glob('/home/alex/pyenv/nlp/news/*.csv')

op_data = [d for d in d_path if re.search(r'cnn', d)][0]
op_df = pd.read_csv(op_data)

f_train, f_test, l_train, l_test = train_test_split(
op_df['title'], op_df['com_lab_bin'], test_size=.2, random_state=2
)

train_label = np.array(l_train)
test_label = np.array(l_test)

#### basic sequential

vocab = len(set(op_df['title'].str.cat().split()))
print(f'vocab size is {vocab}')
vocab_size_mod = 10000
embedding_dim = int(.25*vocab)
pad = 15
max_strlength = max(op_df['title'].apply(lambda n: len(n.split()))) + 15
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<00V>'

tokenizer = Tokenizer(vocab_size_mod, oov_token = '00V')
tokenizer.fit_on_texts(f_train)
word_index = tokenizer.word_index
train_seq = tokenizer.texts_to_sequences(f_train)
train_pad = pad_sequences(train_seq, maxlen = max_strlength,
                          padding = 'post', truncating = 'post')

test_seq = tokenizer.texts_to_sequences(f_test)
test_pad = pad_sequences(test_seq, maxlen = max_strlength,
                         padding = 'post', truncating = 'post') 

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size_mod, embedding_dim, input_length = max_strlength),
tf.keras.layers.GlobalAveragePooling1D(),
tf.keras.layers.Dense(6, activation = 'relu'),
tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
nepochs = 20
history = model.fit(train_pad, train_label, epochs = nepochs,
                   validation_data = (test_pad, test_label))

model.evaluate(test_pad, test_label)

#### sequential RNN

feature, label = op_df['title'].values, op_df['com_lab_bin'].values
x_train, x_test, y_train, y_test = train_test_split(feature, label,
                                                   stratify = label)


vocab = len(set(op_df['title'].str.cat().split()))
print(f'vocab size is {vocab}')
vocab_size = 1000
embed_dim = int(.25*vocab_size)
max_len = max(op_df['title'].apply(lambda n: len(n.split()))) + 5


tokenizer = Tokenizer(vocab_size_mod, oov_token = '00V')
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

train_seq = tokenizer.texts_to_sequences(x_train)
train_pad = pad_sequences(train_seq, maxlen = max_len,
                         padding = 'post')

test_seq = tokenizer.texts_to_sequences(x_test)
test_pad = pad_sequences(test_seq, maxlen = max_len,
                        padding = 'post')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
    input_dim = vocab_size,
    output_dim = embed_dim,
    input_length = max_len,
    mask_zero = True
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(1e-4),
              metrics = ['accuracy'])
model.summary()
n_epochs = 50
history = model.fit(train_pad, y_train,
                    epochs = n_epochs,
                    validation_data = (test_pad, y_test))

# over fitting problem

model.evaluate(test_pad, y_test)
test_samp = ['The outrage of the public']
samp_pad = pad_sequences(tokenizer.texts_to_sequences(test_samp),
                        maxlen = max_len,
                        padding = 'post')

model.predict(samp_pad)