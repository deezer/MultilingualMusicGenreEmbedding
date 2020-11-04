# Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation

This repository provides Python code to reproduce the cross-lingual music genre translation experiments from the article **Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation** presented at the [ISMIR 2020](https://ismir.github.io/ISMIR2020/) conference.

The projects consists of three parts:
- `mmge/data_preparation`: collect and prepare data required for learning music genre embeddings and for evaluation (see [Data preparation](#data-preparation) for more details).
- `mmge/embeddings_learning`: learn English-language only or multilingual music genre embeddings (see [Music genre embedding](#music-genre-embedding) for more details).
- `mmge/tag_translation`: perform and evaluate cross-source English-language only and cross-lingual music genre translation (see [Music genre translation](#music-genre-translation) for more details).

We currently support three languages:
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
We further explain how to reproduce the results reported in *Table 3* and *Table 4* of the article.

### Download data
Data collected from DBpedia, namely the parallel corpus and music genre graph, could change over time. Consequently, we provide for download the version used in the paper experiments. We also include the pre-computed music genre embeddings. More details about how to prepare the data and learn embeddings from scratch can be found in [Data preparation](#data-preparation) and [Music genre embedding](#music-genre-embedding) respectively.

For the cross-source English-language music genre translation, we rely on the same parallel corpus as in our previous work [Leveraging knowledge bases and parallel annotations for music genre translation](https://arxiv.org/abs/1907.08698). We also provide for download the pre-computed translation tables required by the baseline translator. For more information about how these tables are generated, please consult [this git repository](https://github.com/deezer/MusicGenreTranslation).

The data is available [for download on Zenodo](). After download, the `data` folder must be placed in the root folder containing the cloned code. Otherwise, the constant `DATA_DIR` defined in `mmge/utils/utils.py` should be changed accordingly.

The `data` folder contains the following data:
- `[fr|es|en]_entities.txt`: music artists, works and bands from DBpedia in the respective language identified by its code.
- `musical_items_ids.csv`: mapping of DBpedia-based music items on unique identifiers.
- `filtered_musical_items.csv`: the parallel corpus containing DBpedia-based music items with music genre annotations in at least two languagues. This corpus has been filtered, by removing the music genres which did not appear at least 16 times.
- `filtered_dbp_graph.graphml`:
- `folds`:
- `tries`:
- `graphs`:
- `generated_embeddings`:
- `ismir2019baseline`:

### Experiments

To evaluate the cross-lingual music genre translation (*Table 3* in the paper), use `compute_multilingual_results.py` as follows:
```bash
cd mmge/tag_translation/
python compute_multilingual_results.py --target fr
python compute_multilingual_results.py --target es
python compute_multilingual_results.py --target en
```

To evaluate the cross-source English-language music genre translation (*Table 4* in the paper), use `compute_acousticbrainz_results.py` as follows:
```bash
cd mmge/tag_translation/
python compute_acousticbrainz_results.py --target fr
python compute_acousticbrainz_results.py --target es
python compute_acousticbrainz_results.py --target en
```

The target language / source is explicitly specified through the argument `--target`. The translation then happens from the other two languages / sources to the target.

## Run pipeline from scratch

### Data preparation

*Step 1* - collect artists, bands and music works from DBpedia:
```bash
cd mmge/data_preparation
python data_preparation/step1_collect_dbp_music_items.py
```

*Step 2* - collect the music genres annotations of the previously queried music items:
```bash
python data_preparation/step2_collect_dbp_genres_for_music_items.py
```

*Step 3* - filter the corpus by removing music genres that do not appear at least 16 times:
```bash
python data_preparation/step3_filter_corpus.py
```

*Step 4* - split corpus in 4 folds for each language:
```bash
python data_preparation/step4_prepare_folds_eval.py
```

*Step 5* - collect the multilingual DBpedia music genre graph:
```bash
python data_preparation/step5_collect_dbp_genre_graph.py
```

*Step 6* - clean the raw DBpedia graph:
```bash
python data_preparation/step6_clean_dbp_graph.py
```

*Step 7* - create tries to tokenize new tags, potentially written without spaces, in DBpedia-based words:
```bash
python data_preparation/step7_create_tries.py
```

*Step 8* - generate normalized undirected genre graphs for the 2 experiments (multilingual and English-language only):
```bash
python data_preparation/step8_generate_norm_genre_graphs.py
```
For the English-language only experiment, create a new music genre graph from the English DBpedia and the [AcousticBrainz](https://multimediaeval.github.io/2018-AcousticBrainz-Genre-Task/) taxonomies (lastfm, discogs and tagtraum)
Important: make sure that acousticbrainz folder containing the stats files for each taxonomy is downloaded and positioned in the data/ folder (e.g. discogs.csv.stats)

### Music genre embedding

*Step 1* - download [fastText word embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) for English, French and Spanish.

*Step 2* - align French and Spanish embeddings to the English ones by following [these instructions](https://github.com/facebookresearch/fastText/tree/master/alignment),
The aligned embeddings should be saved in the folder `data/aligned_embeddings/`

*Step 3* - generate multilingual music genre embeddings with multiple strategies (see paper):
```
python embeddings_learning/learn_multilingual_embeddings.py multilingual
```

*Step 4* - generate English-language only music genre embeddings with multiple strategies (see paper)
(DBpedia + AcousticBrainz taxonomies)
```
python embeddings_learning/learn_multilingual_embeddings.py acousticbrainz
```

*Step 5* - generate English-language only music genre embeddings with multiple strategies (see paper)
(DBpedia + AcousticBrainz taxonomies)
```
python embeddings_learning/learn_multilingual_embeddings.py multilingual
```

### Music genre translation

The experiments should be run in the same way as for reproducing the published results (see [Experiments](#experiments) above).

The macro-AUC scores may not be identical to the ones reported in the paper because DBpedia could change over time. New music artists, works or bands could appear in DBpedia or some of the past ones could be removed. The annotations of music items with music genres could be modified too. Hence, these changes have an impact on the parallel corpus.

Additionally, the music genre graph could also evolve because music genres or music genre relations are added to or removed from DBpedia over time.

However, we should still reach the same conclusions as presented in the paper:
- Exploiting the semantics of the music genre graph edges leads to marginally improved results w.r.t. the
original retrofitting in the English-language multi-source translation and significantly higher AUC scores in the
cross-lingual translation.
- The initialization with smooth inverse frequency averaging yields better translations than the initialization based on the common average.
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
