# Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation

This repository provides Python code to reproduce the cross-lingual music genre translation experiments presented in the article **Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation** (published in the [ISMIR 2020](https://ismir.github.io/ISMIR2020/) conference).

The projects consists of three parts:
- `mmge/data_preparation`: collect and prepare data required for learning music genre embeddings and for evaluation.
- `mmge/embeddings_learning`: learn English-language only or multilingual music genre embeddings.
- `mmge/tag_translation`: perform and evaluate cross-source English-language only and cross-lingual music genre translation.

For the cross-source English-language only music genre translation, we compare the translation using the new embeddings with the baseline proposed in our [ISMIR 2018 publication](https://github.com/deezer/MusicGenreTranslation), *Leveraging knowledge bases and parallel annotations for music genre translation*.

For the cross-lingual music genre translation, we currently support three languages:
- :gb: English (en)
- :fr: French (fr)
- :es: Spanish (es)

## Installation

```bash
git clone https://github.com/deezer/MultilingualMusicGenreEmbedding
cd MultilingualMusicGenreEmbedding
python setup.py install
```

Requirements: numpy, pandas, sklearn, networkx, joblib, torch, SPARQLWrapper.

## Reproduce published results

### Data

### Experiments

For cross-lingual music genre translation, we have:
```bash
cd mmge/tag_translation/
python compute_multilingual_results.py --target fr
python compute_multilingual_results.py --target es
python compute_multilingual_results.py --target en
```

For cross-source English-language music genre translation, we have:
```bash
cd mmge/tag_translation/
python compute_acousticbrainz_results.py --target fr
python compute_acousticbrainz_results.py --target es
python compute_acousticbrainz_results.py --target en
```

The target language / source is explicitly specified through the argument `--target`. The translation then happens from the other two languages / sources to the target.

## Run experiments from scratch

### Data

### Music genre embeddings

### Music genre translation

The experiments should be run in the same way as for reproducing the published results (see [Experiments](###experiments) above).

The macro-AUC scores may not be identical to the ones reported in the paper because the data collected from DBpedia could change in time. For instance, new musical artists, works or bands could appear in DBpedia or some of the past ones could be removed. The annotations of musical items with music genres could be modified too. Hence, these have an impact on the parallel corpus.
Additionally, the music genre graph could be different too because music genres or music genre relations are added to or removed from DBpedia.

However, we should still reach the same conclusions as presented in the paper:
- Exploiting the semantics of the music genre graph edges leads to marginally improved results w.r.t. the
original retrofitting in the English-language multi-source translation and significantly higher AUC scores in the
cross-lingual translation.
- The sif initialization yields better translations than the avg initialization.
- We outperform the baselines by large margins in both experiments.

# Cite

Please cite our paper if you use this code in your own work:

```BibTeX
@inproceedings{epure2020multilingual,
  title={Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation},
  author={Epure, Elena V. and Salha, Guillaume and Hennequin, Romain},
  booktitle={21st International Society for Music Information Retrieval Conference (ISMIR)},
  year={2020}
}
```
