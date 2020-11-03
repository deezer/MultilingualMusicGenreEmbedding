import os
import csv
import sys
import networkx as nx
from os import listdir

from mmge.utils import utils
from mmge.utils.tag_manager import TagManager
from mmge.utils import trie
sys.modules['trie'] = trie

opj = os.path.join


def init_graph_from_DBpedia(G, langs, lemma_tries):
    """ Create a graph starting from the DBpedia multilingual multidigraph
        :param langs: only the languages in langs are considered
        :param lemma_tries: used in node normalization
        :return: the normalized graph
    """
    # Use a simple undirected graph (original graph is multidigraph)
    newG = nx.Graph()

    # Dict mapping the nodes to their normalized form
    norm_nodes = {}
    # Add nodes to the new graph
    for genre in G:
        lang = genre[:2]
        if lang not in langs:
            continue
        norm_genre = TagManager.normalize_tag_wtokenization(genre, lemma_tries[lang], prefixed=True, asList=False)
        norm_genre = ''.join([lang, ":", norm_genre])
        newG.add_node(norm_genre)
        norm_nodes[genre] = norm_genre

    # Add edges to the new graph
    for u, v, a in G.subgraph(norm_nodes.keys()).edges(data=True):
        newG.add_edge(norm_nodes[u], norm_nodes[v], type=a['type'])
    return newG


def update_graph_with_acbrainztaxs(G, acbrainz_dir, lemma_tries):
    """ Update graph with the nodes and edges from acoustic brainz data
        :param G: the graph to be updated
        :param acbrainz_dir: the folder where info about the AcousticBrainz taxonomies is found
        :param lemma_tries: used in node normalization
        :return: the updated graph
    """
    in_files = [opj(acbrainz_dir, f) for f in listdir(acbrainz_dir)]
    for file in in_files:
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                genres = row[0].split('---')
                norm_g1 = 'en:' + TagManager.normalize_tag_wtokenization(genres[0], lemma_tries['en'], prefixed=False, asList=False)
                if len(genres) == 1:
                    G.add_node(norm_g1)
                else:
                    norm_g2 = 'en:' + TagManager.normalize_tag_wtokenization(genres[1], lemma_tries['en'], prefixed=False, asList=False)
                    G.add_edge(norm_g2, norm_g1, type='musicSubgenre')
                    #print(norm_g1, norm_g2)
    return G


# Load original graph
G = utils.get_graph()
# Load tries
lemma_tries = {}
for lang in utils.langs:
    lemma_tries[lang] = utils.load_trie(lang)

out_dir = ''.join([utils.DATA_DIR, 'graphs/'])
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print("Directory ", out_dir, " Created ")
else:
    print("Directory ", out_dir, " already exists")

# Create normalized multilingual graph and save it
norm_multiling_graph = init_graph_from_DBpedia(G, utils.langs, lemma_tries)
nx.write_graphml(norm_multiling_graph, utils.NORM_MULTILING_GRAPH_PATH)

# Create normalized English graph (DBpedia + AcousticBrainz) and save it
norm_en_graph = init_graph_from_DBpedia(G, ['en'], lemma_tries)
norm_en_graph = update_graph_with_acbrainztaxs(norm_en_graph, utils.ACOUSTICBRAINZ_DIR, lemma_tries)
nx.write_graphml(norm_en_graph, utils.NORM_EN_GRAPH_PATH)

