# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os

import kaggle

MODEL_DIR = "../data/models"
MODEL_ARCH = "qa-lstm.json"
MODEL_WEIGHTS = "qa-lstm-model-best.hdf5"

DATA_DIR = "../data/comp_data"
QA_TRAIN_FILE = "8thGr-NDMC-Train.csv"

WORD2VEC_BIN = "GoogleNews-vectors-negative300.bin.gz"
WORD2VEC_EMBED_SIZE = 300

LSTM_SEQLEN = 196 # from original model
NUM_CHOICES = 4   # number of choices for multiple choice

#### Load up the vectorizer
qapairs = kaggle.get_question_answer_pairs(
    os.path.join(DATA_DIR, QA_TRAIN_FILE))
word2idx = kaggle.build_vocab([], qapairs)
vocab_size = len(word2idx) + 1 # include mask character 0

#### Load up the model
with open(os.path.join(MODEL_DIR, MODEL_ARCH), "rb") as fjson:
    json = fjson.read()
model = model_from_json(json)
model.load_weights(os.path.join(MODEL_DIR, MODEL_WEIGHTS))

#### read in the data ####
#### correct_answer = "B"
question = "Which is a distinction between an epidemic and a pandemic?"
answers = ["the symptoms of the disease",
           "the geographical area affected",
           "the species of organisms infected",
           "the season in which the disease spreads"]
qwords = nltk.word_tokenize(question)
awords_list = [nltk.word_tokenize(answer) for answer in answers]
Xq, Xa = [], []
for idx, awords in enumerate(awords_list):
    Xq.append([word2idx[qword] for qword in qwords])
    Xa.append([word2idx[aword] for aword in awords])
Xq = pad_sequences(Xq, maxlen=LSTM_SEQLEN)
Xa = pad_sequences(Xa, maxlen=LSTM_SEQLEN)

#model.compile(optimizer="adam", loss="categorical_crossentropy",
#              metrics=["accuracy"])
model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
Y = model.predict([Xq, Xa])

# calculate the softmax
probs = np.exp(1.0 - (Y[:, 1] - Y[:, 0]))
probs = probs / np.sum(probs)

plt.bar(np.arange(len(probs)), probs)
plt.xticks(np.arange(len(probs))+0.35, ["A", "B", "C", "D"])
plt.xlabel("choice (x)")
plt.ylabel("probability p(x)")
plt.show()

