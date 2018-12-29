# PyText
Text Classification suite.

## Requirements

### Dataset

[Download](https://users.dcc.uchile.cl/~voyanede/cc5213/datasets/ciperchile_21.12.2018/dataset.zip)

Put content of ```/dataset``` folder contained in the zip into ```ciper_label_generatior/dataset/``` folder.

### Libraries

Do ```pip install -r requirements.txt```

### Word Embeddings

Need spanish pre-trained vectors which can be get [here.](https://github.com/uchile-nlp/spanish-word-embeddings)

#### FastText

[Vector format (.vec.gz) (802 MB)](http://dcc.uchile.cl/~jperez/word-embeddings/fasttext-sbwc.vec.gz)

#### Glove

[Vector format (.vec.gz) (906 MB)](http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz)

#### Word2Vec

[Vector format (.txt.bz2)](http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.txt.bz2)