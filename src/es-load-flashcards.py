# -*- coding: utf-8 -*-
from __future__ import division, print_function
import elasticsearch
import nltk
import os

DATA_DIR = "../data/comp_data"
STORY_FILE = "studystack_qa_cleaner_no_qm.txt"
STORY_INDEX = "flashcards-idx"

es = elasticsearch.Elasticsearch(hosts=[{
    "host": "localhost",
    "port": "9200"
}])

if es.indices.exists(STORY_INDEX):
    print("deleting index: %s" % (STORY_INDEX))
    resp = es.indices.delete(index=STORY_INDEX)
    print(resp)

body = {
    "settings": {
        "number_of_shards": 5,
        "number_of_replicas": 0
    }
}
print("creating index: %s" % (STORY_INDEX))
resp = es.indices.create(index=STORY_INDEX, body=body)
print(resp)

fstory = open(os.path.join(DATA_DIR, STORY_FILE), "rb")
lno = 1
for line in fstory:
    if lno % 1000 == 0:
        print("# stories read: %d" % (lno))
    line = line.strip()
    line = line.decode("utf8").encode("ascii", "ignore")
    fcid, sent, ans = line.split("\t")
    story = " ".join(nltk.word_tokenize(" ".join([sent, ans])))
    doc = { "story": story }
    resp = es.index(index=STORY_INDEX, doc_type="stories", id=lno, body=doc)
#    print(resp["created"])
    lno += 1
print("# stories read and indexed: %d" % (lno))
fstory.close()
es.indices.refresh(index=STORY_INDEX)

query = """ { "query": { "match_all": {} } }"""
resp = es.search(index=STORY_INDEX, doc_type="stories", body=query)
print("# of records in index: %d" % (resp["hits"]["total"]))