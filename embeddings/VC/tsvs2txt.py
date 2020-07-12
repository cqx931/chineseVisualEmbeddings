import tsv
import json
import numpy as np
import csv
import argparse

def generateTxtFromMetas(pathToEmbeddings, pathToMeta, outputPath):
    reader = tsv.TsvReader(open(pathToEmbeddings))
    meta = tsv.TsvReader(open(pathToMeta))
    chara = []
    label = []

    for zi,line in meta:
        label.append(zi);

    label = label[1:]
    # print(len(label))
    with open(outputPath, 'w') as f:
        for count, embedding in enumerate(reader):
            em = list(embedding)
            if count == 0:
                size = len(em)
                f.write(str(len(label)) +  " " + str(size) + "\n")
            data =  " ".join(em)
            if (count < len(label)):
                f.write(label[count] +  " " + data + "\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, help="path to embeddings", default="embeddings/VC/embeddings.tsv")
    parser.add_argument("--meta", type=str, help="path to meta file", default="embeddings/VC/meta.tsv")
    parser.add_argument("--output", type=str, help="output file name", default="embeddings/VC/embeddings_word2vec.txt")
    args = parser.parse_args()

    generateTxtFromMetas(args.embeddings, args.meta, args.output)
    print("Generate word2vec format embeddings to: ", args.output)
