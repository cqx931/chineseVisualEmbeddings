# -*- coding: utf-8 -*-
"""
@author: cqx931
2019
"""

import tools
import sys
sys.setrecursionlimit(10000)

def chainVisualSimilarity(seed,history,limit=50):
    results = tools.getSimilarResult(model, seed, n=limit)
    idx = -1
    if history == "":
        history += seed

    while True:
      idx += 1
      if idx < len(results) :
          newText = results[idx][0]
      elif len(history) < 4743:
          print("Limit", limit+50,len(history) )
          limit = limit + 50
          break
      else:
          break

      if newText not in history:
        break
    history += newText
    if len(history) % 100 == 0:
        print(len(history))

    if len(history) < 4744:
        chainVisualSimilarity(newText, history, limit=limit)
    else:
        print(len(history), history)

def testRelational():
    result = model.most_similar('暖')
    print(result)
    result = model.most_similar(negative=['火'])
    print(result)
    result2 = model.most_similar(positive=['体', '本'], negative=['固'])
    print(result2)

if __name__ == "__main__":
    # model = tools.loadModel('../embeddings/VC/visual_embedding_v3.txt')
    # print(tools.getSimilarResult(model, '朴'))
    # testSimilarByWord('爱')
    # testSimilarByWord('可')
    # testSimilarByWord('非')
    # model = tools.loadModel('../embeddings/VC/visual_embedding_final_v2.txt')
    # # print(tools.getSimilarResult(model, '朴'))
    # print(tools.getSimilarResult(model, '閃'))
    # print(tools.getSimilarResult(model, '正'))
    # print(tools.getSimilarResult(model, '下'))
    # print(tools.getSimilarResult(model, '一'))
    #
    # print("---------")
    #
    # model = tools.loadModel('../embeddings/VC/visual_embedding_final_v2.1.txt')
    # # print(tools.getSimilarResult(model, '朴'))
    # print(tools.getSimilarResult(model, '閃'))
    # print(tools.getSimilarResult(model, '正'))
    # print(tools.getSimilarResult(model, '下'))
    # print(tools.getSimilarResult(model, '一'))

    # model = tools.loadModel('../embeddings/VC/visual_embeddings_subset_v2.txt')
    # # print(tools.getSimilarResult(model, '朴'))
    # print(tools.getSimilarResult(model, '風'))
    # print(tools.getSimilarResult(model, '正'))
    # print(tools.getSimilarResult(model, '下'))
    # print(tools.getSimilarResult(model, '一'))

    model = tools.loadModel('../embeddings/VC/subset/visual_embedding_subset_v2.1.txt')
    print(tools.getAnnoyIndex(model, '乙'));
    # chainVisualSimilarity("字","")
    # print(tools.getSimilarResult(model, '愛'))
    # print(tools.getSimilarResult(model, '風'))
    # print(tools.getSimilarResult(model, '正'))

    # print(tools.getSimilarResult(model, '下'))
    # print(tools.getSimilarResult(model, '一'))

    # model = tools.loadModel('../embeddings/VC/visual_embedding_7351.txt')
    # print(tools.getSimilarResult(model, '朴'))
else:
    model = tools.loadModel('../embeddings/VC/visual_embedding_final_v2.2.txt')
    subset = tools.loadModel('../embeddings/VC/visual_embedding_subset_v2.2.txt')
    print("VISUAL Model loaded")
