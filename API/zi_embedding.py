# -*- coding: utf-8 -*-
"""
@author: cqx931
2019
"""
import tools

def testVocab():
    word = "情"
    print(tools.getSimilarResult(model, word,vocab=5000))
    print(tools.getSimilarResult(model, word,vocab=3000))

if __name__ == "__main__":
    model = tools.loadModel('sgns/sgns.sikuquanshu.bigram.txt')
    print(tools.testSimilarByWord(model, '品'))
    # testVocab()

else:
    model = tools.loadModel('zi/zi_embeddingsgns.literature.bigram-char.txt')
    print("LIT Model loaded")
