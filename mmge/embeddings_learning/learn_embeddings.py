import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from mmge.utils import utils
from mmge.utils import trie
from mmge.utils.tag_manager import TagManager
from retrofit_identity import retrofit_identity


sys.modules['trie'] = trie

opj = os.path.join


def estimated_word_freqs_per_lang(langs, models):
    """Estimate the word frequency from fasttext pre-trained word models
    :param langs: the targetted languages
    :param models: dict of fasttext pre-trained word models per language
    :return: dict of word-frequency estimations per language
    """
    word_ranks = {}
    for lang in langs:
        if lang not in models:
            print('Language not supported')
            return None
        rank = 1
        word_ranks[lang] = {}
        for w in models[lang]:
            word_ranks[lang][w] = rank
            rank += 1
    word_freqs = {}
    for lang in langs:
        word_freqs[lang] = estimate_word_freqs(word_ranks[lang], True)
    return word_freqs


def estimate_word_freqs(word_ranks, mandelbrot=False):
    """ The words are given in descending order
        z = Zipf rank in a list of words ordered by decreasing frequency
        f(z, N) = frequency of a word with Zipf rank z in a list of N words
        f(z, N) = approx. 1/z
        p(word with z rank) = f(z, N) / N = 1/(z * N)
        :param word_ranks: dict of word ranks
        :param mandelbrot: if Mandelbrot generalization of Zipf should be used. Then f(z, N) = 1/(z + 2.7) (Word Frequency Distributions By R. Harald Baayen)
        :return: dict of word frequencies
    """
    word_freqs = {}
    for w in word_ranks:
        if mandelbrot:
            word_freqs[w] = 1 / (word_ranks[w] + 2.7)
        else:
            word_freqs[w] = 1 / word_ranks[w]
    return word_freqs


def generate_initial_embs(emb_type):
    """Generate initial music genre embeddings from aligned fastText
    :param emb_type: the embedding type, average or weighted average
    :return: the initial embeddings and the music genres which are known, meaning they have non-zero initial embeddings
    """
    def _get_emb_avg(g, lang):
        """Compute the embedding of g as the average of its word embeddings
        :param g: the input genre
        :param lang: language
        :return: the embedding and if all words of this genre are known
        """
        emb = np.zeros(emb_dims[lang])
        known_words_count = 0
        words = g.split()
        for w in words:
            if w in models[lang]:
                emb += models[lang][w]
                known_words_count += 1
        emb /= len(words)
        return emb, known_words_count > 0

    def _get_emb_wavg(g, lang, a=0.001):
        """Compute the embeddings of g with a sentence embedding algorithm (average weighted by the word estimated frequencies)
        :param g: the input genre
        :param lang: language
        :param a: a model hyper-parameter (see Arora et al. in the paper)
        :return: the embedding and if all words of this genre are known
        """
        emb = np.zeros(emb_dims[lang])
        known_words_count = 0
        words = g.split()
        for w in words:
            if w in models[lang]:
                emb += a / (a + word_freqs[lang][w]) * models[lang][w]
                known_words_count += 1
        emb /= len(words)
        return emb, known_words_count > 0

    def _remove_pc(df_embs, npc=1):
        """Remove the pc (see Arora at el. in the paper)
        :param df_embs: the input embeddings
        :return: the normalized embeddings
        """
        pc = _compute_pc(df_embs, npc)
        if npc == 1:
            df_embs_out = df_embs - df_embs.dot(pc.transpose()) * pc
        else:
            df_embs_out = df_embs - df_embs.dot(pc.transpose()).dot(pc)
        return df_embs_out

    def _compute_pc(df_embs, npc=1):
        """Compute the pc (see Arora at el. in the paper)
        :param df_embs: the input embeddings
        :return: the principal component
        """
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(df_embs)
        return svd.components_

    embs = {}
    known = {}
    for g in G.nodes:
        lang = g[:2]
        norm_g = TagManager.normalize_tag_wtokenization(g, tries[lang], prefixed=True)
        if emb_type == 'avg':
            embs[g], known[g] = _get_emb_avg(norm_g, lang)
        else:
            embs[g], known[g] = _get_emb_wavg(norm_g, lang)

    embs = pd.DataFrame(embs).T  # the embeddings are columns
    if emb_type == 'sif':  # the algorithm imposes a normalization
        norm_embs = _remove_pc(embs.to_numpy())
        embs = pd.DataFrame(norm_embs, columns=embs.columns, index=embs.index)
    return embs, known


def get_undirected_edges(mapping, G):
    """Return list of undirected edges to be use in retrofitting
    :param G: the music genre graph
    :param mapping: dict that maps tags on unique integers
    :return: a dict of undirected edges per edge type
    """
    edge_types = utils.rels_types
    edges = {}
    for et in edge_types:
        edges[et] = {}
        for g in G.nodes:
            edges[et][mapping[g]] = []
    for s, t, meta in G.edges(data=True):
        #print(s, t)
        edges[meta['type']][mapping[s]].append(mapping[t])
        edges[meta['type']][mapping[t]].append(mapping[s])
    return edges


def beta_f(i, j, edges):
    """Proposed beta function to be used in retrofitting
    :param i: source node
    :param j: target node
    :param edges: list of edges grouped by edge type
    :return: 1 if between i and j there is an equivalence relation type, else 1/degree of i
    """
    ordered_types_by_priority = ['sameAs', 'wikiPageRedirects', 'musicSubgenre', 'stylisticOrigin', 'musicFusionGenre', 'derivative']
    for et in ordered_types_by_priority:
        if et not in edges:
            continue
        if j in edges[et][i]:
            r = et
            break

    if r in utils.equiv_rels_types:
        return 1
    else:
        # Count the number of nodes with the other relations
        count = 0
        for et in ordered_types_by_priority:
            if et not in edges:
                continue
            count += len(edges[et][i])
        return 1 / count if count > 0 else 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: generate_embeddings.py acousticbrainz|multilingual")
        sys.exit(1)

    experiment_name = sys.argv[1]
    if experiment_name == 'multilingual':
        langs = utils.langs
        graph_file = utils.NORM_MULTILING_GRAPH_PATH
    else:
        langs = ['en']
        graph_file = utils.NORM_EN_GRAPH_PATH

    # Prepare the output folder
    out_dir = opj(utils.DATA_DIR, 'generated_embeddings/' + experiment_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print("Directory ", out_dir, "created ")
    else:
        print("Directory ", out_dir, "already exists")

    # Load the aligned fastText word embeddings
    print('Loading aligned fastText word embeddings')
    models = {}
    emb_dims = {}
    for lang in langs:
        models[lang], emb_dims[lang]  = utils.read_embeddings(opj(utils.ALIGNED_FT_EMB_PATH, ''.join(['cc.', lang, '-en.vec'])))
        print(lang, ' loaded')

    # Estimate word frequencies
    print('Estimating word frequencies')
    word_freqs = estimated_word_freqs_per_lang(langs, models)

    # Read the graph
    print("Loading the graph", graph_file)
    G = utils.get_graph(graph_file)

    # Load the tries
    tries = {}
    for lang in langs:
        tries[lang] = utils.load_trie(lang)

    # Learn and save the embeddings of music genres
    print("Learning embeddings")
    for emb_type in utils.emb_composition_types:
        initial_embs, known = generate_initial_embs(emb_type)
        initial_embs.to_csv(opj(out_dir, emb_type + "_initial_embs.csv"), index_label='words')

        index = initial_embs.index.tolist()
        mapping = dict(zip(index, list(range(len(initial_embs.index)))))
        known_mapped = {}
        for k in known:
            known_mapped[mapping[k]] = known[k]
        undirected_edges = get_undirected_edges(mapping, G)

        y_unweighted = retrofit_identity(initial_embs.values, undirected_edges, known_mapped, beta=None, alpha=None, verbose=True)
        retro_all_embs = pd.DataFrame(y_unweighted, index=index)
        retro_all_embs.to_csv(opj(out_dir, emb_type + "_retro_unweighted_embs.csv"), index_label='words')

        y_weighted = retrofit_identity(initial_embs.values, undirected_edges, known_mapped, beta=beta_f, alpha=None, verbose=True)
        retro_all_embs = pd.DataFrame(y_weighted, index=index)
        retro_all_embs.to_csv(opj(out_dir, emb_type + "_retro_weighted_embs.csv"), index_label='words')
