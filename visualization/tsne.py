# -*- coding: utf-8 -*-
import argparse

import gensim
import numpy as np
import csv
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.font_manager as font_manager
import matplotlib as mpl

# font
path_font = '../fonts/NotoSansCJKtc-Regular.ttf'
prop = font_manager.FontProperties(fname=path_font)
mpl.rcParams['font.family'] = "Noto Sans CJK TC"

# displayText = "../data/VC/radicals/water.txt"
# with open(displayText, 'r') as file:
#     subData = file.read().replace('\n', '')

# various sizes
A1 = (33.1, 23.4)
A4 = (11.7, 8.3)
Screen = (16, 10)

def annotateAll(ax, X_tsne, labelsPath):
    labels = []
    with open(labelsPath) as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      for row in reader:
        if row[0] is not "character": # skip headers
            labels.append(row[0])

    ax.DefaultTextFontSize = 7
    max = len(X_tsne[:, 0])

    for i,label in enumerate(labels):
        # if label in subData:
        if i < max:
            ax.text(X_tsne[:, 0][i], X_tsne[:, 1][i], label)
            return
    return

def viz(embeddings_path, output_filename, labels_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=False)
    X = model.vectors

    print("model loaded", len(X))

    X_tsne = TSNE(n_components=2, perplexity=20, early_exaggeration=12, learning_rate=100).fit_transform(X)
    print("tsne", len(X_tsne[:, 0]))
    # set figure size, dpi and no black

    fig = figure(num=None, figsize=Screen, dpi=72, frameon=False)
    # set bg color
    fig.patch.set_facecolor('#FFFFFF')

    ax = fig.add_axes([0, 0, 1, 1])
    # hide axis
    ax.axis('off')
    # wo = white oval, ko = black oval
    ax.plot(X_tsne[:, 0], X_tsne[:, 1], 'ko', ms=72./fig.dpi*1, alpha=0.8)
    # annotateAll(ax, X_tsne, labels_path);
    fig.savefig(output_filename, facecolor=fig.get_facecolor(), edgecolor='none')

def main(args):
    viz(args.embeddings, args.output, args.labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, help="path to embeddings file", default="../embeddings/VC/v3.2/v3.2_embeddings.txt")
    parser.add_argument("--output", type=str, help="output file name", default="tsne_visualization.png")
    parser.add_argument("--labels", type=str, help="path to labels file", default="../embeddings/VC/v3.2/v3.2_proj_meta.tsv")
    args = parser.parse_args()
    main(args)
