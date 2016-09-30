# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import os

import babi

BABI_DIR = "../data/babi_data/tasks_1-20_v1-2/en"
TASK_NBR = 1
EMBED_HIDDEN_SIZE = 50
BATCH_SIZE = 32
NBR_EPOCHS = 40

train_file, test_file = babi.get_files_for_task(TASK_NBR, BABI_DIR)

data_train = babi.get_stories(os.path.join(BABI_DIR, train_file))
data_test = babi.get_stories(os.path.join(BABI_DIR, test_file))

word2idx = babi.build_vocab([data_train, data_test])
vocab_size = len(word2idx) + 1
print("vocab_size=", vocab_size)

story_maxlen, question_maxlen = babi.get_maxlens([data_train, data_test])
print("story_maxlen=", story_maxlen)
print("question_maxlen=", question_maxlen)

Xs_train, Xq_train, Y_train = babi.vectorize(data_train, word2idx, 
                                             story_maxlen, question_maxlen)
Xs_test, Xq_test, Y_test = babi.vectorize(data_test, word2idx,
                                          story_maxlen, question_maxlen)
print(Xs_train.shape, Xq_train.shape, Y_train.shape)
print(Xs_test.shape, Xq_test.shape, Y_test.shape)

# define model
# generate embeddings for stories
story_rnn = Sequential()
story_rnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                        input_length=story_maxlen))
story_rnn.add(Dropout(0.3))

# generate embeddings for question and make adaptable to story
question_rnn = Sequential()
question_rnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                           input_length=question_maxlen))
question_rnn.add(Dropout(0.3))
question_rnn.add(LSTM(EMBED_HIDDEN_SIZE, return_sequences=False))
question_rnn.add(RepeatVector(story_maxlen))

# merge the two
model = Sequential()
model.add(Merge([story_rnn, question_rnn], mode="sum"))
model.add(LSTM(EMBED_HIDDEN_SIZE, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", 
              metrics=["accuracy"])

print("Training...")
model.fit([Xs_train, Xq_train], Y_train, 
          batch_size=BATCH_SIZE, nb_epoch=NBR_EPOCHS, validation_split=0.05)
loss, acc = model.evaluate([Xs_test, Xq_test], Y_test, batch_size=BATCH_SIZE)
print()
print("Test loss/accuracy = {:.4f}, {:.4f}".format(loss, acc))
