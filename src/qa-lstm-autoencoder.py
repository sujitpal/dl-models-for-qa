# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.layers import Input, RepeatVector
from keras.models import Model
from keras.layers.recurrent import LSTM
from sklearn.cross_validation import train_test_split
import numpy as np
import os

import kaggle

DATA_DIR = "../data/comp_data"
QA_TRAIN_FILE = "8thGr-NDMC-Train.csv"
STORY_FILE = "studystack_qa_cleaner_no_qm.txt"
STORY_WEIGHTS = "lstm-story-weights.txt"
STORY_BIAS = "lstm-story-bias.txt"

EMBED_SIZE = 64
BATCH_SIZE = 256
NBR_EPOCHS = 20

stories = kaggle.get_stories(os.path.join(DATA_DIR, STORY_FILE))
story_maxlen = max([len(words) for words in stories])

# this part is only required to get the maximum sequence length
qapairs = kaggle.get_question_answer_pairs(
    os.path.join(DATA_DIR, QA_TRAIN_FILE))
question_maxlen = max([len(qapair[0]) for qapair in qapairs])
answer_maxlen = max([len(qapair[1]) for qapair in qapairs])
seq_maxlen = max([story_maxlen, question_maxlen, answer_maxlen])

word2idx = kaggle.build_vocab(stories, qapairs)
vocab_size = len(word2idx)

Xs = kaggle.vectorize_stories(stories, word2idx, seq_maxlen)
Xstrain, Xstest = train_test_split(Xs, test_size=0.3, random_state=42)
print(Xstrain.shape, Xstest.shape)

inputs = Input(shape=(seq_maxlen, vocab_size))
encoded = LSTM(EMBED_SIZE)(inputs)
decoded = RepeatVector(seq_maxlen)(encoded)
decoded = LSTM(vocab_size, return_sequences=True)(decoded)
autoencoder = Model(inputs, decoded)

autoencoder.compile("adadelta", loss="binary_crossentropy")

autoencoder.fit(Xstrain, Xstrain, nb_epoch=NBR_EPOCHS, batch_size=BATCH_SIZE,
                shuffle=True, validation_data=(Xstest, Xstest))

# save weight matrix for embedding (transforms from seq_maxlen to EMBED_SIZE)
weight_matrix, bias_vector = autoencoder.layers[1].get_weights()
np.savetxt(os.path.join(DATA_DIR, STORY_WEIGHTS), weight_matrix)
np.savetxt(os.path.join(DATA_DIR, STORY_BIAS), bias_vector)
