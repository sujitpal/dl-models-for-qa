# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model
from sklearn.cross_validation import train_test_split
import os
import sys

import kaggle

DATA_DIR = "../data/comp_data"
MODEL_DIR = "../data/models"
WORD2VEC_BIN = "studystack.bin"
WORD2VEC_EMBED_SIZE = 300

QA_TRAIN_FILE = "8thGr-NDMC-Train.csv"
QA_TEST_FILE = "8thGr-NDMC-Test.csv"

QA_EMBED_SIZE = 64
BATCH_SIZE = 128
NBR_EPOCHS = 20

## extract data
print("Loading and formatting data...")
qapairs = kaggle.get_question_answer_pairs(
    os.path.join(DATA_DIR, QA_TRAIN_FILE))
question_maxlen = max([len(qapair[0]) for qapair in qapairs])
answer_maxlen = max([len(qapair[1]) for qapair in qapairs])

# Even though we don't use the test set for classification, we still need
# to consider any additional vocabulary words from it for when we use the
# model for prediction (against the test set).
tqapairs = kaggle.get_question_answer_pairs(
    os.path.join(DATA_DIR, QA_TEST_FILE), is_test=True)    
tq_maxlen = max([len(qapair[0]) for qapair in tqapairs])
ta_maxlen = max([len(qapair[1]) for qapair in tqapairs])

seq_maxlen = max([question_maxlen, answer_maxlen, tq_maxlen, ta_maxlen])

word2idx = kaggle.build_vocab([], qapairs, tqapairs)
vocab_size = len(word2idx) + 1 # include mask character 0

Xq, Xa, Y = kaggle.vectorize_qapairs(qapairs, word2idx, seq_maxlen)
Xqtrain, Xqtest, Xatrain, Xatest, Ytrain, Ytest = \
    train_test_split(Xq, Xa, Y, test_size=0.3, random_state=42)
print(Xqtrain.shape, Xqtest.shape, Xatrain.shape, Xatest.shape, 
      Ytrain.shape, Ytest.shape)

# get embeddings from word2vec
print("Loading Word2Vec model and generating embedding matrix...")
embedding_weights = kaggle.get_weights_word2vec(word2idx,
    os.path.join(DATA_DIR, WORD2VEC_BIN), is_custom=True)
        
print("Building model...")

# output: (None, QA_EMBED_SIZE, seq_maxlen)
qin = Input(shape=(seq_maxlen,), dtype="int32")
qenc = Embedding(input_dim=vocab_size,
                 output_dim=WORD2VEC_EMBED_SIZE,
                 input_length=seq_maxlen,
                 weights=[embedding_weights])(qin)
qenc = LSTM(QA_EMBED_SIZE, return_sequences=True)(qenc)
qenc = Dropout(0.3)(qenc)

# output: (None, QA_EMBED_SIZE, seq_maxlen)
ain = Input(shape=(seq_maxlen,), dtype="int32")
aenc = Embedding(input_dim=vocab_size,
                 output_dim=WORD2VEC_EMBED_SIZE,
                 input_length=seq_maxlen,
                 weights=[embedding_weights])(ain)
aenc = LSTM(QA_EMBED_SIZE, return_sequences=True)(aenc)
aenc = Dropout(0.3)(aenc)

# attention model
attn = merge([qenc, aenc], mode="dot", dot_axes=[1, 1])
attn = Flatten()(attn)
attn = Dense(seq_maxlen * QA_EMBED_SIZE)(attn)
attn = Reshape((seq_maxlen, QA_EMBED_SIZE))(attn)

qenc_attn = merge([qenc, attn], mode="sum")
qenc_attn = Flatten()(qenc_attn)

output = Dense(2, activation="softmax")(qenc_attn)

model = Model(input=[qin, ain], output=[output])

print("Compiling model...")
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

print("Training...")
best_model_filename = os.path.join(MODEL_DIR, 
    kaggle.get_model_filename(sys.argv[0], "best"))
checkpoint = ModelCheckpoint(filepath=best_model_filename,
                             verbose=1, save_best_only=True)
model.fit([Xqtrain, Xatrain], [Ytrain], batch_size=BATCH_SIZE,
          nb_epoch=NBR_EPOCHS, validation_split=0.1,
          callbacks=[checkpoint])

print("Evaluation...")
loss, acc = model.evaluate([Xqtest, Xatest], [Ytest], batch_size=BATCH_SIZE)
print("Test loss/accuracy final model = %.4f, %.4f" % (loss, acc))

final_model_filename = os.path.join(MODEL_DIR, 
    kaggle.get_model_filename(sys.argv[0], "final"))
json_model_filename = os.path.join(MODEL_DIR,
    kaggle.get_model_filename(sys.argv[0], "json"))
kaggle.save_model(model, json_model_filename, final_model_filename)

best_model = kaggle.load_model(json_model_filename, best_model_filename)
best_model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
loss, acc = best_model.evaluate([Xqtest, Xatest], [Ytest], batch_size=BATCH_SIZE)
print("Test loss/accuracy best model = %.4f, %.4f" % (loss, acc))
   
