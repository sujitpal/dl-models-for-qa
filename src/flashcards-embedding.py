# -*- coding: utf-8 -*-
from __future__ import division, print_function
from gensim.models.word2vec import Word2Vec, LineSentence
import logging
import multiprocessing
import nltk
import os

DATA_DIR = "../data/comp_data"
FLASHCARD_SENTS = "studystack_qa_cleaner_no_qm.txt"
FLASHCARD_MODEL = "studystack.bin"
EMBED_SIZE = 300  # so we can reuse code using word2vec embeddings

logger = logging.getLogger("flashcards-embedding")
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")
logging.root.setLevel(level=logging.DEBUG)

class FlashcardSentences(object):
    def __init__(self, filename):
        self.filename = filename
        
    def __iter__(self):
        for line in open(self.filename, "rb"):
            line = line.strip()
            line = line.decode("utf8").encode("ascii", "ignore")
            _, question, answer = line.split("\t")
            qwords = nltk.word_tokenize(question)
            awords = nltk.word_tokenize(answer)
            yield qwords + awords

# build model from sentences (CBOW w/negative sampling)
model = Word2Vec(size=EMBED_SIZE, window=5, min_count=5,
                 workers=multiprocessing.cpu_count())
sentences = FlashcardSentences(os.path.join(DATA_DIR, FLASHCARD_SENTS))
model.build_vocab(sentences)
sentences = FlashcardSentences(os.path.join(DATA_DIR, FLASHCARD_SENTS))
model.train(sentences)
                 
model.init_sims(replace=True)

model.save(os.path.join(DATA_DIR, FLASHCARD_MODEL))

# test model
model = Word2Vec.load(os.path.join(DATA_DIR, FLASHCARD_MODEL))
print(model.similarity("man", "woman"), model.similarity("cat", "rock"))
print(model.most_similar("exercise"))