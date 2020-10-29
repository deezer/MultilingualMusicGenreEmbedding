import os
import numpy as np

from tag_translation.judge import Judge
from tag_translation.tag_manager import TagManager
from tag_translation.data_helper import DataHelper

from tag_translation.baseline_translators import DbpMappingTranslator
from tag_translation.translators import EnglishLangEmbsTranslator

import utils


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", required=True)
    args = parser.parse_args()
    target = args.target
    sources = list({"lastfm", "discogs", "tagtraum"} - {target})
    dataset_path = os.path.join(utils.FOLDS_DIR, "{0}_4-fold_by_artist.tsv".format(target))
    dhelper = DataHelper(sources, target, dataset_path=dataset_path)
    print(dhelper.dataset_df)

    source_tags = [f"{source}:{el}" for source in sources for el in utils.corpus_genres_for_source(dhelper.dataset_df, source)]
    target_tags = [f"{target}:{el}" for el in utils.corpus_genres_for_source(dhelper.dataset_df, target)]
    tm = TagManager.get(sources, target, source_tags, target_tags)
    judge = Judge()
    print("The judged was initialized")

    translators = {}
    tr = DbpMappingTranslator(tm, "/Users/eepure/Documents/Projects/TagTranslationResearch/data/test_data/distance_table_dbpedia_" + target)
    translators['baseline'] = tr

    models = {}
    models['avg_init'] = "data/generated_embeddings/acousticbrainz/avg_initial_embs.csv"
    models['avg_retro_unweighted'] = "data/generated_embeddings/acousticbrainz/avg_retro_unweighted_embs.csv"
    models['avg_retro_weighted'] = "data/generated_embeddings/acousticbrainz/avg_retro_weighted_embs.csv"
    models['sif_init'] = "data/generated_embeddings/acousticbrainz/sif_initial_embs.csv"
    models['sif_retro_unweighted'] = "data/generated_embeddings/acousticbrainz/sif_retro_unweighted_embs.csv"
    models['sif_retro_weighted'] = "data/generated_embeddings//acousticbrainz/sif_retro_weighted_embs.csv"

    for k in models:
        print("Initializing model", k)
        translators[k] = EnglishLangEmbsTranslator(tm, models[k], ['en'])

    print("Evaluating the translators")
    for k in translators:
        print(k)
        tr = translators[k]
        results = []
        for fold in range(4):
            eval_data, eval_target = dhelper.get_test_data(tm, fold=fold)
            eval_target = eval_target.astype("float32")
            print("Computing KB results for fold {}".format(fold))
            res = judge.compute_macro_metrics(eval_target, tr.predict_scores(eval_data))
            print(res)
            results.append(res)
        print('mean ', np.mean(results))
        print('std ', np.std(results))
