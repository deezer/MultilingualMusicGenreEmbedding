import os
import pandas as pd
import networkx as nx

from utils import utils

opj = os.path.join
graph_file = utils.RAW_GRAPH_PATH
corpus_file = utils.CORPUS_FILE_PATH

# Read corpus
df = pd.read_csv(corpus_file, index_col='id')
all_genres = utils.all_formatted_genres(df)

# Read raw DBpedia music genre graph
G = nx.read_graphml(graph_file)
clean_G = nx.Graph(G)

# The first cleaning step consists in removing those nodes for which the correct pattern for entity naming does not apply
renamed_nodes = {}
for g in G.nodes():
    lang = g[:2]
    # To get the entity name, a regular expression is used. If it does not apply then the genre does not exist as a proper DBpedia resource
    g_name = utils.get_ent_name(g)
    if g_name is None:
        clean_G.remove_node(g)
        print(g)
    else:
        renamed_nodes[g] = ''.join([lang, ":", g_name])
print(len(G.nodes()) - len(clean_G.nodes()), " nodes were removed")
clean_G = nx.relabel_nodes(clean_G, renamed_nodes)

# The second cleaning step is to remove the connected components which do not contain corpus music genres
ccs = sorted(nx.connected_components(clean_G), key=len)
for c in ccs:
    intersect = all_genres.intersection(c)
    if len(intersect) == 0:
        clean_G.remove_nodes_from(c)

nx.write_graphml(clean_G, utils.GRAPH_PATH)
