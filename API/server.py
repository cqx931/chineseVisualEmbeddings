"""
@author: cqx931
2019
"""

from flask import Flask
from flask_cors import CORS
from flask import request
import json
from hanziconv import HanziConv

import tools
import visual_embedding

app = Flask(__name__)

cors = CORS(app, resources={r"/chineseEmbeddings": {"origins": "*"}})

@app.route("/")
def hello():
    return "Chinese Embeddings:v1.0.1"

@app.route('/chineseEmbeddings', methods=['GET'])
# @cross_origin(origin='*',headers=['Content- Type','Authorization'])

def getEmbeddings():

    z = request.args.get('target')
    t = request.args.get('type')
    n = request.args.get('topn')
    vocab = request.args.get('vocab')

    if n == None: n = 10

    result = []

    if t == 'visual':
        print("[visual]")
        result = tools.getSimilarResult(visual_embedding.model, z)
    elif t == 'commonVisual':
        print("[common visual]")
        result = tools.getSimilarResult(visual_embedding.subset, z)
    elif t == 'semantics':
        print("[semantics]")
        result = tools.getSimilarResult(zi_embedding.model, z)
    elif t == 'visualSemantics':

        def common_member(a, b):
            a = [i[0] for i in a]
            b = [i[0] for i in b]

            # multiply result from ZiEmbeddings to traditional ch
            allB = ''.join(b)
            traditionalB = list(HanziConv.toTraditional(allB))
            b = b + traditionalB

            a_set = set(a)
            b_set = set(b)
            if (a_set & b_set):
                return list(a_set & b_set)
            else:
                print(a,b)
                return []

        print("[visual semantics]")
        result1 = tools.getSimilarResult(visual_embedding.model, z, n=100)
        result2 = tools.getSimilarResult(zi_embedding.model, z, n=100)

        print("1:", result1)
        print("2:", result2)
        result = common_member(result1, result2)
        print("result:", result)

    print(result)

    data = {
    "type": t,
    "target": z,
    "result": result
    }

    return json.dumps(data)

if __name__ == "__main__":
    app.run(host="138.16.162.17", port=8084)
