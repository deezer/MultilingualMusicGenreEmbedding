import os
import numpy as np
import networkx as nx

from judge import Judge
from tag_manager import TagManager
from data_helper import DataHelper
from baseline_translators import GraphDistanceMapper
from translators import MultilingualEmbsTranslator

from mmge.utils import utils

opj = os.path.join


if __name__ == "__main__":
    from datetime import datetime
    startTime = datetime.now()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", required=True)
    args = parser.parse_args()
    target = args.target
    sources = list(set(utils.langs) - {target})

    dataset_path = os.path.join(utils.FOLDS_DIR, "{0}_4-fold.tsv".format(target))
    dhelper = DataHelper(sources, target, dataset_path=dataset_path)
    for lang in utils.langs:
        dhelper.dataset_df[lang] = dhelper.dataset_df[lang].apply(lambda l: [utils.get_ent_name(t) for t in l])
    print(dhelper.dataset_df)

    source_tags = [f"{source}:{el}" for source in sources for el in utils.corpus_genres_for_source(dhelper.dataset_df, source)]
    target_tags = [f"{target}:{el}" for el in utils.corpus_genres_for_source(dhelper.dataset_df, target)]
    tm = TagManager.get(sources, target, source_tags, target_tags)
    judge = Judge()
    print("The judged was initialized")

    translators = {}
    G = nx.read_graphml(utils.NORM_MULTILING_GRAPH_PATH)
    translators['baseline'] = GraphDistanceMapper(tm, G, utils.langs)

    models = {}
    models['avg_init'] = ''.join([utils.MULTILING_EMBS_DIR, "avg_initial_embs.csv"])
    models['avg_retro_unweighted'] = ''.join([utils.MULTILING_EMBS_DIR, "avg_retro_unweighted_embs.csv"])
    models['avg_retro_weighted'] = ''.join([utils.MULTILING_EMBS_DIR, "avg_retro_weighted_embs.csv"])
    models['sif_init'] = ''.join([utils.MULTILING_EMBS_DIR, "sif_initial_embs.csv"])
    models['sif_retro_unweighted'] = ''.join([utils.MULTILING_EMBS_DIR, "sif_retro_unweighted_embs.csv"])
    models['sif_retro_weighted'] = ''.join([utils.MULTILING_EMBS_DIR, "sif_retro_weighted_embs.csv"])
    for k in models:
        print("Initializing model", k)
        translators[k] = MultilingualEmbsTranslator(tm, models[k], utils.langs)

    judge = Judge()
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
            print('auc_macro', utils.truncate(res * 100, 1))
            results.append(res)
        print('auc_macro mean', utils.truncate(np.mean(results) * 100, 1))
        print('auc_macro std', utils.truncate(np.std(results) * 100, 1))

    print(datetime.now() - startTime)
