# Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation

This repository provides Python code to reproduce the cross-lingual music genre translation experiments presented in the article **Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation** (published in the [ISMIR 2020](https://ismir.github.io/ISMIR2020/) conference).

The projects consists of three parts:
- `mmge/data_preparation`: collect and prepare data required for learning music genre embeddings and for evaluation.
- `mmge/embeddings_learning`: learn English-language only or multilingual music genre embeddings.
- `mmge/tag_translation`: perform and evaluate cross-source English-language only and cross-lingual music genre translation.

For the cross-source English-language only music genre translation, we compare the translation using the new embeddings with the baseline proposed in our [ISMIR 2018 publication](https://github.com/deezer/MusicGenreTranslation), *Leveraging knowledge bases and parallel annotations for music genre translation*.

For the cross-lingual music genre translation, we currently support three languages:
- :gb: English,
- :fr: French,
- :es: Spanish.

## Installation

```bash
python setup.py install
```

Requirements: numpy, pandas, sklearn, networkx, joblib, torch, SPARQLWrapper.

## Cite

Please cite our paper if you use this code in your own work:

```BibTeX
@inproceedings{epure2020multilingual,
  title={Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation},
  author={Epure, Elena V. and Salha, Guillaume and Hennequin, Romain},
  booktitle={21st International Society for Music Information Retrieval Conference (ISMIR)},
  year={2020}
}
```
