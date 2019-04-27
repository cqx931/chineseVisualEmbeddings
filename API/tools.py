import gensim
import operator

def loadModel(path):
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    return model

def sort(result):
    return sorted(dict(result).items(), key=operator.itemgetter(1), reverse=True)

def getSimilarResult(model, word, n=20, vocab=None):
    return sort(model.similar_by_word(word, topn=n,restrict_vocab=vocab))

def testSimilarByWord(word):
    print("----" + word + "----")
    print(sort(ve.similar_by_word(word)))
