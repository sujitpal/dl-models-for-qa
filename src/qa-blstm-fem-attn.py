# -*- coding: utf-8 -*-
from __future__ import division, print_function
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Merge, Dropout, Reshape, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
import numpy as np
import os

import kaggle

DATA_DIR = "../data/comp_data"
MODEL_DIR = "../data/models"
WORD2VEC_BIN = "studystack.bin"
WORD2VEC_EMBED_SIZE = 300

QA_TRAIN_FILE = "8thGr-NDMC-Train.csv"

QA_EMBED_SIZE = 64
BATCH_SIZE = 32
NBR_EPOCHS = 20

## extract data

print("Loading and formatting data...")
qapairs = kaggle.get_question_answer_pairs(
    os.path.join(DATA_DIR, QA_TRAIN_FILE))
question_maxlen = max([len(qapair[0]) for qapair in qapairs])
answer_maxlen = max([len(qapair[1]) for qapair in qapairs])
seq_maxlen = max([question_maxlen, answer_maxlen])

word2idx = kaggle.build_vocab([], qapairs)
vocab_size = len(word2idx) + 1 # include mask character 0

Xq, Xa, Y = kaggle.vectorize_qapairs(qapairs, word2idx, seq_maxlen)
Xqtrain, Xqtest, Xatrain, Xatest, Ytrain, Ytest = \
    train_test_split(Xq, Xa, Y, test_size=0.3, random_state=42)
print(Xqtrain.shape, Xqtest.shape, Xatrain.shape, Xatest.shape, 
      Ytrain.shape, Ytest.shape)

# get embeddings from word2vec
# see https://github.com/fchollet/keras/issues/853
print("Loading Word2Vec model and generating embedding matrix...")
word2vec = Word2Vec.load(os.path.join(DATA_DIR, WORD2VEC_BIN))
embedding_weights = np.zeros((vocab_size, WORD2VEC_EMBED_SIZE))
for word, index in word2idx.items():
    try:
        embedding_weights[index, :] = word2vec[word.lower()]
    except KeyError:
        pass  # keep as zero (not ideal, but what else can we do?)
        
print("Building model...")
qenc = Sequential()
qenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=seq_maxlen,
                   weights=[embedding_weights]))
qenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True), 
                       merge_mode="sum"))
qenc.add(Dropout(0.3))

aenc = Sequential()
aenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=seq_maxlen,
                   weights=[embedding_weights]))
aenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True),
                       merge_mode="sum"))
aenc.add(Dropout(0.3))

# attention model
attn = Sequential()
attn.add(Merge([qenc, aenc], mode="dot", dot_axes=[1, 1]))
attn.add(Flatten())
attn.add(Dense((seq_maxlen * QA_EMBED_SIZE)))
attn.add(Reshape((seq_maxlen, QA_EMBED_SIZE)))

model = Sequential()
model.add(Merge([qenc, attn], mode="sum"))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

print("Training...")
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "qa-blstm-fem-attn-best.hdf5"),
    verbose=1, save_best_only=True)
model.fit([Xqtrain, Xatrain], Ytrain, batch_size=BATCH_SIZE,
          nb_epoch=NBR_EPOCHS, validation_split=0.1,
          callbacks=[checkpoint])

print("Evaluation...")
loss, acc = model.evaluate([Xqtest, Xatest], Ytest, batch_size=BATCH_SIZE)
print("Test loss/accuracy final model = %.4f, %.4f" % (loss, acc))

model.save_weights(os.path.join(MODEL_DIR, "qa-blstm-fem-attn-final.hdf5"))
with open(os.path.join(MODEL_DIR, "qa-blstm-fem-attn.json"), "wb") as fjson:
    fjson.write(model.to_json())

model.load_weights(filepath=os.path.join(MODEL_DIR, "qa-blstm-fem-attn-best.hdf5"))
loss, acc = model.evaluate([Xqtest, Xatest], Ytest, batch_size=BATCH_SIZE)
print("\nTest loss/accuracy best model = %.4f, %.4f" % (loss, acc))
