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
    # print(sort(lit.similar_by_word(word,topn=20,restrict_vocab=5000)))

if __name__ == "__main__":

    # Load Google's pre-trained Word2Vec model.
    # context_word_char = loadModel('zi/zi_embedding.context.word-character.char1-1.txt')
    # target_word_char = loadModel('zi/zi_embeddingsgns.target.word-character.char1-1.dynwin5.thr10.neg5.dim300.iter5.txt')
    # target_ngram = loadModel('zi/zi_embeddings.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5.txt')
    # model = tools.loadModel('zi/zi_embeddingsgns.literature.bigram-char.txt')
    model = tools.loadModel('sgns/sgns.sikuquanshu.bigram.txt')
    #
    print(tools.testSimilarByWord(model, '品'))
   
    # testVocab()

else:
    model = tools.loadModel('zi/zi_embeddingsgns.literature.bigram-char.txt')
    print("LIT Model loaded")
