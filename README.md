# Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation

This repository provides Python code to reproduce the cross-lingual music genre translation experiments presented in the article **Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation** ([ISMIR 2020](https://ismir.github.io/ISMIR2020/) conference).

The projects consists of three parts:
- `mmge/data_preparation`: collect and prepare data required for learning music genre embeddings and for evaluation.
- `mmge/embeddings_learning`: learn English-language only or multilingual music genre embeddings.
- `mmge/tag_translation`: reproduce translation experiments with multi-source English-language only music genres (for more details see our [ISMIR 2018](https://github.com/deezer/MusicGenreTranslation) publication, *Leveraging knowledge bases and parallel annotations for music genre translation*) or with multilingual music genres (currently, three languages are supported :gb: English, :fr: French and :es: Spanish).

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
