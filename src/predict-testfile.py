# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np
import os

import kaggle

DATA_DIR = "../data/comp_data"
TRAIN_FILE = "8thGr-NDMC-Train.csv"
TEST_FILE = "8thGr-NDMC-Test.csv"
SUBMIT_FILE = "submission.csv"

MODEL_DIR = "../data/models"
MODEL_JSON = "qa-lstm-fem-attn.json"
MODEL_WEIGHTS = "qa-lstm-fem-attn-best.h5"
LSTM_SEQLEN = 196 # seq_maxlen from original model

print("Loading model..")
model = kaggle.load_model(os.path.join(MODEL_DIR, MODEL_JSON),
                          os.path.join(MODEL_DIR, MODEL_WEIGHTS))
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

print("Loading vocabulary...")
qapairs = kaggle.get_question_answer_pairs(os.path.join(DATA_DIR, TRAIN_FILE))
tqapairs = kaggle.get_question_answer_pairs(os.path.join(DATA_DIR, TEST_FILE), 
                                            is_test=True)
word2idx = kaggle.build_vocab([], qapairs, tqapairs)
vocab_size = len(word2idx) + 1 # include mask character 0

ftest = open(os.path.join(DATA_DIR, TEST_FILE), "rb")
fsub = open(os.path.join(DATA_DIR, SUBMIT_FILE), "wb")
fsub.write("id,correctAnswer\n")
line_nbr = 0
for line in ftest:
    line = line.strip().decode("utf8").encode("ascii", "ignore")
    if line.startswith("#"):
        continue
    if line_nbr % 10 == 0:
        print("Processed %d questions..." % (line_nbr))
    cols = line.split("\t")
    qid = cols[0]
    question = cols[1]
    answers = cols[2:]
    # create batch of question
    qword_ids = [word2idx[qword] for qword in nltk.word_tokenize(question)]
    Xq, Xa = [], []
    for answer in answers:
        Xq.append(qword_ids)
        Xa.append([word2idx[aword] for aword in nltk.word_tokenize(answer)])
    Xq = pad_sequences(Xq, maxlen=LSTM_SEQLEN)
    Xa = pad_sequences(Xa, maxlen=LSTM_SEQLEN)
    Y = model.predict([Xq, Xa])
    probs = np.exp(1.0 - (Y[:, 1] - Y[:, 0]))
    correct_answer = chr(ord('A') + np.argmax(probs))
    fsub.write("%s,%s\n" % (qid, correct_answer))
    line_nbr += 1
print("Processed %d questions..." % (line_nbr))
fsub.close()
ftest.close()
