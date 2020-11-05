# Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation

This repository provides Python code to reproduce the cross-lingual music genre translation experiments from the article **Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation** presented at the [ISMIR 2020](https://ismir.github.io/ISMIR2020/) conference.

The projects consists of three parts:
- `mmge/data_preparation`: collect and prepare data required for learning music genre embeddings and for evaluation (see [Data preparation](#data-preparation) for more details).
- `mmge/embeddings_learning`: learn English-language only or multilingual music genre embeddings (see [Music genre embedding](#music-genre-embedding) for more details).
- `mmge/tag_translation`: perform and evaluate cross-source English-language only or cross-lingual music genre translation (see [Music genre translation](#music-genre-translation) for more details).

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

Requirements: numpy, pandas, sklearn, networkx, joblib, torch, SPARQLWrapper, spacy.

## Reproduce published results
We further explain how to reproduce the results reported in *Table 3* and *Table 4* of the article.

### Download data
Data collected from DBpedia, namely the parallel corpus and music genre graph, could change over time. Consequently, we provide for download the version used in the paper experiments. We also include the pre-computed music genre embeddings. More details about how to prepare the data and learn embeddings from scratch can be found in [Data preparation](#data-preparation) and [Music genre embedding](#music-genre-embedding) respectively.

For the cross-source English-language music genre translation, we rely on the same parallel corpus as in our previous work [Leveraging knowledge bases and parallel annotations for music genre translation](https://arxiv.org/abs/1907.08698). We also provide for download the pre-computed distance tables required by the baseline translator. For more information about how these tables are generated, please consult [this git repository](https://github.com/deezer/MusicGenreTranslation).

The data is available [for download on Zenodo](). After download, the `data` folder must be placed in the root folder containing the cloned code. Otherwise, the constant `DATA_DIR` defined in `mmge/utils/utils.py` should be changed accordingly.

The `data` folder contains the following data:
- `acousticbrainz`: the English-language music genre taxonomies released in the [MediaEval 2018 AcousticBrainz Genre Task](https://mtg.github.io/acousticbrainz-genre-dataset/data/).
- `[fr|es|en]_entities.txt`: music artists, works and bands from DBpedia in the language identified by the code.
- `musical_items_ids.csv`: mapping of DBpedia-based music items on unique identifiers.
- `filtered_musical_items.csv`: the multilingual parallel corpus containing DBpedia-based music items with music genre annotations in at least two languagues. This corpus has been filtered by removing music genres which did not appear at least 16 times (to ensure that each music genre appears 4 times in each of the 4 folds).
- `filtered_dbp_graph.graphml`: the multilingual DBpedia-based music genre graph in a cleaned version. Tags that were not recognized as proper DBpedia resources and the connected components that did not contain at least a corpus music genre were removed.
- `folds`: the parallel corpus split in 4 folds in a stratified way for each source / language as target.
- `tries`: deserialized `Trie` (`mmge/utils/trie.py`) objects for each language created from vocabularies extracted from DBpedia music genre tags.
- `graphs`: graphs with normalized tags as nodes for each experiment, English-language only and multilingual.
- `generated_embeddings`: English-language only and multilingual music genre embeddings learned with various strategies to initialize the embeddings and different retrofitting versions.
- `ismir2019baseline`: pre-computed distance tables to be used by the baseline translator in the cross-source English-language translation experiments.

### Experiments

Evaluate music genre embeddings in cross-lingual music genre translation (*Table 3* in the paper):
```bash
cd mmge/tag_translation/
python compute_multilingual_results.py --target fr
python compute_multilingual_results.py --target es
python compute_multilingual_results.py --target en
```

Evaluate music genre embeddings in cross-source English-language music genre translation (*Table 4* in the paper):
```bash
cd mmge/tag_translation/
python compute_acousticbrainz_results.py --target discogs
python compute_acousticbrainz_results.py --target lastfm
python compute_acousticbrainz_results.py --target tagtraum
```

The target language / source is explicitly specified through the argument `--target`. The translation then happens from the other two languages / sources to the target.

Expected running time:

Target     | `fr` | `es` | `en` | `discogs` | `lastfm` | `tagtraum`|
| :--------: |:--------:|--------:|--------:|--------:|-------:|-------:|
Time    | 4m | 8m | 8m | 1h25m | 1h25m | 1h3m |

## Run pipeline from scratch
We further explain how to run the full pipeline from scratch.
Before starting, multiple resources needed for the evaluation of the English-language only translation baseline should be downloaded from Zenodo (see [Download data](#download-data)) and positioned in the `data` folder: `acousticbrainz`, `ismir2019baseline` and `folds` (only the files `[lastfm|discogs|tagtraum]_4-fold_by_artist.tsv`).

### Data preparation
Each step uses the output of the previous step as input. Therefore, it is important that the previous step finishes correctly. A problem that could appear is that DBpedia in a certain language could be temporarily down. In this case, there are two options:
- wait until DBpedia is again up and could be queried correctly.
- use the data we provided for download to artificially replace the output of the problematic step, thus ensuring the input of the next step. Repeat this for all the problematic steps.

#### Step 1: collect DBpedia music artists, bands and music works
```bash
cd mmge/data_preparation/
python data_preparation/step1_collect_dbp_music_items.py
```
Input: nothing
Output: `[fr|es|en]_entities.txt` and `musical_items_ids.csv`

#### Step 2: collect DBpedia-based music genres annotations for music items
```bash
python data_preparation/step2_collect_dbp_genres_for_music_items.py
```
Input: `[fr|es|en]_entities.txt` and `musical_items_ids.csv`
Output: `musical_items.csv`

#### Step 3: filter the corpus by removing music genres that do not appear at least 16 times
```bash
python data_preparation/step3_filter_corpus.py
```
Input: `musical_items.csv`
Output: `filtered_musical_items.csv`

#### Step 4: split corpus in 4 folds for each language
```bash
python data_preparation/step4_prepare_folds_eval.py
```
Input: `filtered_musical_items.csv`
Output: the files of type `[fr|es|en]_4-fold.tsv`) in the `folds` folder

#### Step 5 collect the multilingual DBpedia-based music genre graph
```bash
python data_preparation/step5_collect_dbp_genre_graph.py
```
Input: `filtered_musical_items.csv`
Output: `dbp_multigraph.graphml`

#### Step 6: clean the raw DBpedia graph
```bash
python data_preparation/step6_clean_dbp_graph.py
```
Input: `dbp_multigraph.graphml`
Output: `filtered_dbp_graph.graphml`

#### Step 7: create tries per language from words of music genres discovered from DBpedia
```bash
python data_preparation/step7_create_tries.py
```
Input: `filtered_dbp_graph.graphml`
Output: the `tries` folder

#### Step 8: generate normalized undirected genre graphs for the 2 experiments (multilingual and English-language only)
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

### Music genre translation

The experiments should be run in the same way as for reproducing the published results (see [Experiments](#experiments) above).

The macro-AUC scores may not be identical to the ones reported in the paper because DBpedia could change over time. New music artists, works or bands could appear in DBpedia or some of the past ones could be removed. The annotations of music items with music genres could be modified too. Hence, these changes have an impact on the parallel corpus.

Additionally, the music genre graph could also evolve because music genres or music genre relations are added to or removed from DBpedia.

However, we should still reach the same conclusions as those presented in the paper:
- Exploiting the semantics of the music genre graph edges leads to marginally improved results w.r.t. the
original retrofitting in the English-language multi-source translation and significantly higher macro-AUC scores in the
cross-lingual translation.
- The initialization with smooth inverse frequency (*sif*) averaging yields better translations than the initialization with the ordinary average (*avg*).
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
