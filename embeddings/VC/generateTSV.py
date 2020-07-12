import pickle
import tsv
import numpy as np
import json
import torch
import argparse

def generateMetaTSVFromData(inputPath, outputPath):

    with open(inputPath) as f:
        data = json.load(f)
    writer = tsv.TsvWriter(open(outputPath, "w"))
    print("Generate meta file for",len(data), " entries")
    # writer.comment("id character class")
    writer.line("character\tclass")
    for id, idx in enumerate(data):
    	#id #char #class
    	item = data[idx]
    	writer.line(item[1] + "\t" + item[2] )
    writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # args.data, args.output
    parser.add_argument("--input", type=str, help="path to data file", default="data/VC/v3.2/v3.2_256C.json")
    parser.add_argument("--output", type=str, help="output file name", default="embeddings/VC/embeddings.tsv")
    args = parser.parse_args()

    # '../../data/VC/final_v2.2.json'
    # "final_v2.2_meta.tsv"

    generateMetaTSVFromData(args.input, args.output)
