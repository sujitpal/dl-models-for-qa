# -*- coding: utf-8 -*-
from __future__ import division, print_function
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Merge, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
import numpy as np
import os

import kaggle

DATA_DIR = "../data/comp_data"
MODEL_DIR = "../data/models"

WORD2VEC_BIN = "GoogleNews-vectors-negative300.bin.gz"
WORD2VEC_EMBED_SIZE = 300

SQA_TRAIN_FILE = "SQA-Train.csv"

QA_EMBED_SIZE = 64
BATCH_SIZE = 32
NBR_EPOCHS = 20

## extract data

print("Loading and formatting data...")
sqatriples = kaggle.get_story_question_answer_triples(
    os.path.join(DATA_DIR, SQA_TRAIN_FILE))
story_maxlen = max([len(sqatriple[0]) for sqatriple in sqatriples])
question_maxlen = max([len(sqatriple[1]) for sqatriple in sqatriples])
answer_maxlen = max([len(sqatriple[2]) for sqatriple in sqatriples])

word2idx = kaggle.build_vocab_from_sqa_triples(sqatriples)
vocab_size = len(word2idx) + 1 # include mask character 0

Xs, Xq, Xa, Y = kaggle.vectorize_sqatriples(sqatriples, word2idx, story_maxlen, 
                                            question_maxlen, answer_maxlen)
Xstrain, Xstest, Xqtrain, Xqtest, Xatrain, Xatest, Ytrain, Ytest = \
    train_test_split(Xs, Xq, Xa, Y, test_size=0.3, random_state=42)
print(Xstrain.shape, Xstest.shape, Xqtrain.shape, Xqtest.shape, 
      Xatrain.shape, Xatest.shape, Ytrain.shape, Ytest.shape)

# get embeddings from word2vec
# see https://github.com/fchollet/keras/issues/853
print("Loading Word2Vec model and generating embedding matrix...")
word2vec = Word2Vec.load_word2vec_format(
    os.path.join(DATA_DIR, WORD2VEC_BIN), binary=True)
embedding_weights = np.zeros((vocab_size, WORD2VEC_EMBED_SIZE))
for word, index in word2idx.items():
    try:
        embedding_weights[index, :] = word2vec[word.lower()]
    except KeyError:
        pass  # keep as zero (not ideal, but what else can we do?)
        
print("Building model...")

# story encoder.
# output shape: (None, story_maxlen, QA_EMBED_SIZE)
senc = Sequential()
senc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=story_maxlen,
                   weights=[embedding_weights], mask_zero=True))
senc.add(LSTM(QA_EMBED_SIZE, return_sequences=True))
senc.add(Dropout(0.3))

# question encoder
# output shape: (None, question_maxlen, QA_EMBED_SIZE)
qenc = Sequential()
qenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=question_maxlen,
                   weights=[embedding_weights], mask_zero=True))
qenc.add(LSTM(QA_EMBED_SIZE, return_sequences=True))                   
qenc.add(Dropout(0.3))

# answer encoder
# output shape: (None, answer_maxlen, QA_EMBED_SIZE)
aenc = Sequential()
aenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=answer_maxlen,
                   weights=[embedding_weights], mask_zero=True))
aenc.add(LSTM(QA_EMBED_SIZE, return_sequences=True))                   
aenc.add(Dropout(0.3))

# merge story and question => facts
# output shape: (None, story_maxlen, question_maxlen)
facts = Sequential()
facts.add(Merge([senc, qenc], mode="dot", dot_axes=[2, 2]))

# merge question and answer => attention
# output shape: (None, answer_maxlen, question_maxlen)
attn = Sequential()
attn.add(Merge([aenc, qenc], mode="dot", dot_axes=[2, 2]))

# merge facts and attention => model
# output shape: (None, story+answer_maxlen, question_maxlen)
model = Sequential()
model.add(Merge([facts, attn], mode="concat", concat_axis=1))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

print("Training...")
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "qa-lstm-story-best.hdf5"),
    verbose=1, save_best_only=True)
model.fit([Xstrain, Xqtrain, Xatrain], Ytrain, batch_size=BATCH_SIZE,
          nb_epoch=NBR_EPOCHS, validation_split=0.1,
          callbacks=[checkpoint])

print("Evaluation")
loss, acc = model.evaluate([Xstest, Xqtest, Xatest], Ytest, 
                           batch_size=BATCH_SIZE)              
print("Test loss/accuracy final model: %.4f, %.4f" % (loss, acc))

model.save_weights(os.path.join(MODEL_DIR, "qa-lstm-story-final.hdf5"))
with open(os.path.join(MODEL_DIR, "qa-lstm-story.json"), "wb") as fjson:
    fjson.write(model.to_json())

model.load_weights(filepath=os.path.join(MODEL_DIR, "qa-lstm-story-best.hdf5"))
loss, acc = model.evaluate([Xstest, Xqtest, Xatest], Ytest, 
                           batch_size=BATCH_SIZE)              
print("Test loss/accuracy best model: %.4f, %.4f" % (loss, acc))
