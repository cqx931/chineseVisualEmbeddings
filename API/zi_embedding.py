# -*- coding: utf-8 -*-
"""
@author: cqx931
"""
import tools

def testSimilarByWord(word):
    print("----" + word + "----")
    print(sort(context_word_char.similar_by_word(word)))
    print(sort(target_word_char.similar_by_word(word)))
    print(sort(target_ngram.similar_by_word(word)))
    print(sort(lit.similar_by_word(word)))
    print(sort(siku.similar_by_word(word)))

def testMostSimilar(word):
    print("----" + word + "----")
    print(context_word_char.most_similar(positive = [word]))
    print(target_word_char.most_similar(positive = [word]))
    print(target_ngram.most_similar(positive = [word]))
    print(lit.most_similar(positive = [word]))
    print(siku.most_similar(positive = [word]))

def testVocab():
    word = "情"
    print(tools.getSimilarResult(model, word,vocab=5000))
    print(tools.getSimilarResult(model, word,vocab=3000))
    # print(sort(lit.similar_by_word(word,topn=20,restrict_vocab=5000)))

if __name__ == "__main__":

    # Load Google's pre-trained Word2Vec model.
    # context_word_char = loadModel('zi/zi_embedding.context.word-character.char1-1.txt')
    # target_word_char = loadModel('zi/zi_embeddingsgns.target.word-character.char1-1.dynwin5.thr10.neg5.dim300.iter5.txt')
    # target_ngram = loadModel('zi/zi_embeddings.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.txt')
    model = tools.loadModel('zi/zi_embeddingsgns.literature.bigram-char.txt')
    # siku = loadModel('zi/sgns.sikuquanshu.bigram.txt')
    #
    # testSimilarByWord('情')
    # testSimilarByWord('打')
    # testSimilarByWord('吃')
    # testSimilarByWord('树')
    # testSimilarByWord('一')
    # testSimilarByWord('道')
    testVocab()

else:
    model = tools.loadModel('zi/zi_embeddingsgns.literature.bigram-char.txt')
    print("LIT Model loaded")

# Based on test result, lit has the best zi vectors. A fetch should not only covers the
# closest but a list of candidate to have more meaningful result.

# print("*****************")

# testMostSimilar('爱')
# testMostSimilar('打')
# testMostSimilar('树')
# testMostSimilar('一')
# testMostSimilar('道')
