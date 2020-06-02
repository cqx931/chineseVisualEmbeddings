# chineseVisualEmbeddings

- Download 'NotoSansCJKsc-Regular' and 'NotoSansCJKtc-Regular' from [Google Noto Fonts](https://www.google.com/get/noto/) and save them to this folder

- install all the packages
`pip3 install -r requirements.txt`
- Preprocessing
preprocess.py
-> final.json

STEP2 Train embeddings
`python CNN.py --data data/VC/final.json --output final_embeddings`

STEP3 generate tsv files
3.1 Generate meta tsv if needed
`python generateTSV.py`
