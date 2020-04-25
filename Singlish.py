import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

with open(
        '/Users/Prakash/Downloads/the-national-university-of-singapore-sms-corpus/smsCorpus_en_2015.03.09_all.json') as f:
    data = json.load(f)

total_words = 10000
oov_token = '<OOV>'


def create_dataset(raw_dataset):
    corpus = []
    for text in raw_dataset['smsCorpus']['message']:
        corpus.append(str(text['text']['$']))

    tokenizer = Tokenizer(num_words=total_words, oov_token=oov_token)
    tokenizer.fit_on_texts(corpus)
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max(len(x) for x in input_sequences)
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    x_train, y_train = input_sequences[:, :-1], input_sequences[:, -1]
    y_train = tf.keras.utils.to_categorical(y_train)

    return x_train, y_train


X_train, Y_train = create_dataset(data)


def Model(x_train, y_train):
    model = tf.keras.Sequential()
    model.add(Embedding(input_dim=total_words, output_dim=64, input_length=x_train.shape[1]))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=20, verbose=1, validation_split=0.1)

    return history
















