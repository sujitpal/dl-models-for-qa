# -*- coding: utf-8 -*-
from __future__ import division, print_function
from gensim.models import Word2Vec
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np
import collections
import os

def get_stories(story_file, debug=False):
    stories = []
    lno = 0
    fin = open(story_file, "rb")
    for line in fin:
        if debug == True and lno % 100 == 0:
            print("# stories read: %d" % (lno))
        line = line.strip()
        line = line.decode("utf8").encode("ascii", "ignore")
        fcid, sent, ans = line.split("\t")
        stories.append(nltk.word_tokenize(" ".join([sent, ans])))
        lno += 1
    fin.close()
    return stories
    
def get_question_answer_pairs(question_file, is_test=False):
    qapairs = []
    fqa = open(question_file, "rb")
    for line in fqa:
        if line.startswith("#"):
            continue
        line = line.strip().decode("utf8").encode("ascii", "ignore")
        cols = line.split("\t")
        question = cols[1]
        qwords = nltk.word_tokenize(question)
        if not is_test:
            correct_ans = cols[2]
            answers = cols[3:]
            # training file parsing
            correct_ans_idx = ord(correct_ans) - ord('A')
            for idx, answer in enumerate(answers):
                awords = nltk.word_tokenize(answer)
                qapairs.append((qwords, awords, idx == correct_ans_idx))
        else:
            # test file parsing (no correct answer)
            answers = cols[2:]
            for answer in answers:
                awords = nltk.word_tokenize(answer)
                qapairs.append((qwords, awords, None))
    fqa.close()
    return qapairs

def get_story_question_answer_triples(sqa_file):
    sqatriples = []
    fsqa = open(sqa_file, "rb")
    for line in fsqa:
        line = line.strip().decode("utf8").encode("ascii", "ignore")
        if line.startswith("#"):
            continue
        story, question, answer, correct = line.split("\t")
        swords = []
        story_sents = nltk.sent_tokenize(story)
        for story_sent in story_sents:
            swords.extend(nltk.word_tokenize(story_sent))
        qwords = nltk.word_tokenize(question)
        awords = nltk.word_tokenize(answer)
        is_correct = int(correct) == 1
        sqatriples.append((swords, qwords, awords, is_correct))
    fsqa.close()
    return sqatriples

def build_vocab(stories, qapairs, testqs):
    wordcounts = collections.Counter()
    for story in stories:
        for sword in story:
            wordcounts[sword] += 1
    for qapair in qapairs:
        for qword in qapair[0]:
            wordcounts[qword] += 1
        for aword in qapair[1]:
            wordcounts[aword] += 1
    for testq in testqs:
        for qword in testq[0]:
            wordcounts[qword] += 1
        for aword in testq[1]:
            wordcounts[aword] += 1
    words = [wordcount[0] for wordcount in wordcounts.most_common()]
    word2idx = {w: i+1 for i, w in enumerate(words)}  # 0 = mask
    return word2idx

def build_vocab_from_sqa_triples(sqatriples):
    wordcounts = collections.Counter()
    for sqatriple in sqatriples:
        for sword in sqatriple[0]:
            wordcounts[sword] += 1
        for qword in sqatriple[1]:
            wordcounts[qword] += 1
        for aword in sqatriple[2]:
            wordcounts[aword] += 1
    words = [wordcount[0] for wordcount in wordcounts.most_common()]
    word2idx = {w: i+1 for i, w in enumerate(words)}  # 0 = mask
    return word2idx

def vectorize_stories(stories, word2idx, story_maxlen):
    Xs = []
    for story in stories:
        Xs.append([word2idx[word] for word in story])
    return pad_sequences(Xs, maxlen=story_maxlen)

def vectorize_qapairs(qapairs, word2idx, seq_maxlen):
    Xq, Xa, Y = [], [], []
    for qapair in qapairs:
        Xq.append([word2idx[qword] for qword in qapair[0]])
        Xa.append([word2idx[aword] for aword in qapair[1]])
        Y.append(np.array([1, 0]) if qapair[2] else np.array([0, 1]))
    return (pad_sequences(Xq, maxlen=seq_maxlen), 
            pad_sequences(Xa, maxlen=seq_maxlen),
            np.array(Y))

def vectorize_sqatriples(sqatriples, word2idx, story_maxlen, 
                         question_maxlen, answer_maxlen):
    Xs, Xq, Xa, Y = [], [], [], []
    for sqatriple in sqatriples:
        Xs.append([word2idx[sword] for sword in sqatriple[0]])
        Xq.append([word2idx[qword] for qword in sqatriple[1]])
        Xa.append([word2idx[aword] for aword in sqatriple[2]])
        Y.append(np.array([1, 0]) if sqatriple[3] else np.array([0, 1]))
    return (pad_sequences(Xs, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=question_maxlen),
            pad_sequences(Xa, maxlen=answer_maxlen),
            np.array(Y))

def get_weights_word2vec(word2idx, w2vfile, w2v_embed_size=300, 
                         is_custom=False):
    word2vec = None
    if is_custom:
        word2vec = Word2Vec.load(w2vfile)
    else:
        word2vec = Word2Vec.load_word2vec_format(w2vfile, binary=True)
    vocab_size = len(word2idx) + 1
    embedding_weights = np.zeros((vocab_size, w2v_embed_size))
    for word, index in word2idx.items():
        try:
            embedding_weights[index, :] = word2vec[word.lower()]
        except KeyError:
            pass  # keep as zero (not ideal, but what else can we do?)
    return embedding_weights

def get_model_filename(caller, model_type):
    caller = os.path.basename(caller)
    caller = caller[0:caller.rindex(".")]
    if model_type == "json":
        return "%s.%s" % (caller, model_type)
    else:
        return "%s-%s.h5" % (caller, model_type)

def save_model(model, json_filename, weights_filename):
    model.save_weights(weights_filename)
    with open(json_filename, "wb") as fjson:
        fjson.write(model.to_json())

def load_model(json_filename, weights_filename):
    with open(json_filename, "rb") as fjson:
        model = model_from_json(fjson.read())
    model.load_weights(filepath=weights_filename)
    return model

    
##### main ####
#
#import os
#
#DATA_DIR = "../data/comp_data"
#QA_TRAIN_FILE = "8thGr-NDMC-Train.csv"
#STORY_FILE = "studystack_qa_cleaner_no_qm.txt"
#
#stories = get_stories(os.path.join(DATA_DIR, STORY_FILE))
#story_maxlen = max([len(words) for words in stories])
#print("story maxlen=", story_maxlen)
#
#qapairs = get_question_answer_pairs(os.path.join(DATA_DIR, QA_TRAIN_FILE))
#question_maxlen = max([len(qapair[0]) for qapair in qapairs])
#answer_maxlen = max([len(qapair[1]) for qapair in qapairs])
#print("q=", question_maxlen, "a=", answer_maxlen)
#
#word2idx = build_vocab(stories, qapairs)
#w2v = get_weights_word2vec(word2idx, 
#                           os.path.join(DATA_DIR, "studystack.bin"),
#                           is_custom=True)
#print(w2v.shape)                           
#
#Xs = vectorize_stories(stories, word2idx, story_maxlen)
#Xq, Xa = vectorize_qapairs(qapairs, word2idx, question_maxlen, answer_maxlen)
