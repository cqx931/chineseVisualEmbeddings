"""
@author: cqx931
2019
"""

import gensim
import operator
from gensim.similarities.index import AnnoyIndexer


def loadModel(path):
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    return model

def sort(result):
    return sorted(dict(result).items(), key=operator.itemgetter(1), reverse=True)

def getSimilarResult(model, word, n=20, vocab=None):
    return sort(model.similar_by_word(word, topn=n, restrict_vocab=vocab))

def getAnnoyIndex(model, word, n=20):
	indexer = AnnoyIndexer(model, 2)
	return model.most_similar(word, topn=n, indexer=indexer)

def testSimilarByWord(model, word):
    print("----" + word + "----")
    print(sort(model.similar_by_word(word)))
