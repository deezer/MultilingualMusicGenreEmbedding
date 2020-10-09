# Instructions to learn music genre embeddings

Packages required:
```
numpy
pandas
sklearn
```

### Step 1: download [fastText word embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) for English, French and Spanish

### Step 2: align French and Spanish embeddings to the English ones by following [these instructions](https://github.com/facebookresearch/fastText/tree/master/alignment). The aligned embeddings should be saved in the folder `data/aligned_embeddings/`

### Step 3: generate multilingual music genre embeddings with multiple strategies as explained in the paper
```
python embeddings_learning/learn_multilingual_embeddings.py multilingual
```

### Step 4: generate English-only music genre embeddings (DBpedia + AcousticBrainz taxonomies) with multiple strategies as explained in the paper
```
python embeddings_learning/learn_multilingual_embeddings.py acousticbrainz
```
