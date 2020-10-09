# Instructions to collect the multilingual corpus and the music genre graph from DBpedia

Packages required:
```
numpy
pandas
networkx
sklearn
SPARQLWrapper
```

### Step 1: collect artists, bands and music works from DBpedia
```
python data_preparation/step1_collect_dbp_music_items.py
```

### Step 2: collect the music genres of the previously collected music items
```
python data_preparation/step2_collect_dbp_genres_for_music_items.py
```

### Step 3: filter the corpus by removing music genres that do not appear at least 16 times
```
python data_preparation/step3_filter_corpus.py
```

### Step 4: split corpus in 4 folds for each language
```
python data_preparation/step4_prepare_folds_eval.py
```

### Step 5: collect the multilingual DBpedia music genre graph
```
python data_preparation/step5_collect_dbp_genre_graph.py
```

### Step 6: clean the raw DBpedia graph
```
python data_preparation/step6_clean_dbp_graph.py
```

### Step 7: create tries to split new tags, potentially written without spaces, in DBpedia-based words
```
python data_preparation/step7_create_tries.py
```

### Step 8: Generate normalized undirected genre graphs for the 2 experiments. For the English only experiment, create a new music genre graph from the English DBpedia and the AcousticBrainz taxonomies (lastfm, discogs and tagtraum)

Important: make sure that acousticbrainz folder containing the stats files for each taxonomy is downloaded and positioned in the data/ folder (e.g. discogs.csv.stats)
```
python data_preparation/step8_generate_norm_genre_graphs.py
```



