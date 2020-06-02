# -*- coding: utf-8 -*-
"""
@author: cqx931
2019
"""

import tools

def testZi():
    testSimilarByWord('湖')

def testCi():
    testSimilarByWord("道路")
    testSimilarByWord("道家")
    testSimilarByWord("道行")

def testVocab():
    word = "月"
    print("-5k-")
    print(tools.getSimilarResult(model, word,vocab=5000))
    print("-10k-")
    print(tools.getSimilarResult(model, word,vocab=10000))
    print("-20k-")
    print(tools.getSimilarResult(model, word,vocab=20000))

if __name__ == "__main__":

    model = tools.loadModel('sgns/sgns.literature.bigram-char.txt')
    print("Lit model loaded")
    # testCi();
    # testZi()
    testVocab()

else:
    model = tools.loadModel('sgns/sgns.literature.bigram-char.txt')
    print("LIT[word] Model loaded")
