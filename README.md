# The Chinese Visual Embeddings
The Chinese Visual Embeddings are trained for the purpose of visual representations for the Chinese written signs.
It is part of a larger project [Graphein](https://chenqianxun.com/graphein/index.html) by Qianxun Chen

### Training Chinese Visual Embeddings

#### STEP 0 Setup the environment

- Download 'NotoSansCJKsc-Regular' and 'NotoSansCJKtc-Regular' from [Google Noto Fonts](https://www.google.com/get/noto/) and save them to folder 'fonts'
- Create a new virtual environment with python 3 `virtualenv --python=python3.6 venv`
- Activate the environment `activate source venv/bin/activate`
- Install all the packages `pip3 install -r requirements.txt`

#### STEP 1 Prepare the data
`python preprocess.py`

#### STEP 2 Train embeddings
- An earlier version of the embeddings was trained with `python CNN.py`
- The latest version of the embeddings can be trained with `python CNN_cuda_multiLabel.py`

#### STEP3 Generate supporting files
- Generate tsv files to preview embeddings in [Embedding Projector](https://projector.tensorflow.org/) `python embeddings/VC/generateTSV.py`
- Generate word2vec format embeddings `python embeddings/VC/tsvs2txt.py`

### API


### Acknowledgement
dict.csv is created based on the [dictionary.txt](https://github.com/skishore/makemeahanzi/blob/master/dictionary.txt) file from the project, [makemehanzi](https://github.com/skishore/makemeahanzi).
