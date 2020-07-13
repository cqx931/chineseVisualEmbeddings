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
    model = tools.loadModel('../embeddings/VC/v3.2/v3.2_embeddings_ep19.txt')
    print(tools.getAnnoyIndex(model2, '情'));

else:
    model = tools.loadModel('../embeddings/VC/v3.2/v3.2_embeddings_ep19.txt')
    print("VISUAL Model loaded")
