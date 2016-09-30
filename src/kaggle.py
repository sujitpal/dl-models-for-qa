# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.preprocessing.sequence import pad_sequences
import nltk
import numpy as np
import collections

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
    
def get_question_answer_pairs(question_file):
    qapairs = []
    fqa = open(question_file, "rb")
    for line in fqa:
        if line.startswith("#"):
            continue
        line = line.strip().decode("utf8").encode("ascii", "ignore")
        qid, question, correct_ans, ans_a, ans_b, ans_c, ans_d = \
            line.split("\t")
        qwords = nltk.word_tokenize(question)
        correct_ans_idx = ord(correct_ans) - ord('A')
        answers = [ans_a, ans_b, ans_c, ans_d]
        for idx, answer in enumerate(answers):
            awords = nltk.word_tokenize(answer)
            qapairs.append((qwords, awords, idx == correct_ans_idx))
    fqa.close()
    return qapairs

def get_story_question_answer_triples(sqa_file):
    sqatriples = []
    fsqa = open(sqa_file, "rb")
    for line in fsqa:
        line = line.strip().decode("utf8").encode("ascii", "ignore")
        story, question, answer, correct = line.split("\t")
        swords = []
        story_sents = nltk.sent_tokenize(story)
        for story_sent in story_sents:
            swords.extend(nltk.word_tokenize(story_sent))
        qwords = nltk.word_tokenize(question)
        awords = nltk.word_tokenize(answer)
        is_correct = int(correct) == 1
        sqatriples.append((swords, qwords, awords, is_correct))
    return sqatriples

def build_vocab(stories, qapairs):
    wordcounts = collections.Counter()
    for story in stories:
        for sword in story:
            wordcounts[sword] += 1
    for qapair in qapairs:
        for qword in qapair[0]:
            wordcounts[qword] += 1
        for aword in qapair[1]:
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

##### main ####
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
#
#Xs = vectorize_stories(stories, word2idx, story_maxlen)
#Xq, Xa = vectorize_qapairs(qapairs, word2idx, question_maxlen, answer_maxlen)

