import os
import time
from string import Template

import networkx as nx
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

from utils import utils
from utils.utils import langs

opj = os.path.join


def get_MusicGenres_ents(lang):
    """Collect entities of type MusicGenre found in DBpedia
    :param lang: the targetted language
    :return: the music genres discovered
    """
    query_template = Template("""SELECT ?genre {{
                SELECT ?genre
                    WHERE {
                        ?genre rdf:type <http://dbpedia.org/ontology/MusicGenre>
                    }
                    ORDER BY ?genre
                }}
                OFFSET $offset
                LIMIT 10000""")
    endpoint = utils.get_endpoint_for_lang(lang)
    sparql_dbpedia = SPARQLWrapper(endpoint + "sparql")
    sparql_dbpedia.setReturnFormat(JSON)
    genres = set()
    offset = 0
    while (True):
        query = query_template.substitute({'offset': offset})
        sparql_dbpedia.setQuery(query)
        results = sparql_dbpedia.query().convert()
        for result in results["results"]["bindings"]:
            genres.add(result["genre"]["value"])
        if len(results["results"]["bindings"]) < 10000:
            break
        offset += 10000
    return genres


def get_MusicGenres_aliases(lang, genres):
    """Collect aliases of the given DBpedia music genres
    :param lang: the targetted language
    :param genres: a list of DBpedia music genres
    :return: the input genres with their aliases
    """
    query_template_alias = Template("""SELECT ?genre, ?alias {{
                SELECT ?genre, ?alias
                    WHERE {
                        ?genre rdf:type <http://dbpedia.org/ontology/MusicGenre>.
                        ?genre owl:sameAs ?alias.
                        FILTER ($other_lang_cond)
                    }
                ORDER BY ?genre
                }}
                OFFSET $offset
                LIMIT 10000""")

    endpoint = utils.get_endpoint_for_lang(lang)
    other_langs_cond = utils.get_alias_filter(lang, langs)
    sparql_dbpedia = SPARQLWrapper(endpoint + "sparql")
    sparql_dbpedia.setReturnFormat(JSON)
    genres_with_aliases = {}
    offset = 0
    while (True):
        query = query_template_alias.substitute({'offset': offset, 'other_lang_cond': other_langs_cond})
        #print(query)
        sparql_dbpedia.setQuery(query)
        results = sparql_dbpedia.query().convert()
        for result in results["results"]["bindings"]:
            genre = result["genre"]["value"]
            if genre not in genres_with_aliases:
                genres_with_aliases[genre] = set()
            alias = result["alias"]["value"]
            genres_with_aliases[genre].add(alias)
            other_lang = utils.get_lang(alias)
            genres[other_lang].add(alias)
        if len(results["results"]["bindings"]) < 10000:
            break
        offset += 10000
    return genres_with_aliases


def collect_genres_from_seeds(genres, lang):
    """Collect new genres by crawling the input DBpedia music genres' relations
    :param genres: a list of DBpedia music genres
    :param lang: the targetted language
    :return: the input genres with their related genres discovered during crawling
    """
    query_template = Template("""SELECT ?property, ?genre2, ?genre1
                  WHERE {
                    ?genre2 ?property ?genre1.
                    FILTER (?genre1 IN ($list)).
                    FILTER (?property IN ($genre_rels))
                  }""")
    query_template_inv = Template("""SELECT ?property, ?genre2, ?genre1
                    WHERE {
                    ?genre1 ?property ?genre2.
                    FILTER (?genre1 IN ($list)).
                    FILTER (?property IN ($genre_rels))
                  }""")

    endpoint = utils.get_endpoint_for_lang(lang)
    sparql_dbpedia = SPARQLWrapper(endpoint + "sparql")
    sparql_dbpedia.setReturnFormat(JSON)
    genre_rels_cond = utils.get_genre_rels_filter(lang)

    seeds = list(genres)
    relations = {}

    start = 0
    while start < len(seeds):
        end = start + 50
        if end > len(seeds):
            end = len(seeds)
        #print(start, end)
        list_genres_str = utils.get_seeds_filter(seeds[start:end])
        for i in range(start, end):
            genres.add(seeds[i])
        start = end

        query = query_template.substitute({'list': list_genres_str, 'genre_rels': genre_rels_cond})
        process_query(query, sparql_dbpedia, relations, seeds, genres)
        query = query_template_inv.substitute({'list': list_genres_str, 'genre_rels': genre_rels_cond})
        process_query(query, sparql_dbpedia, relations, seeds, genres, True)

    return relations


def process_query(query, sparql_dbp, relations, seeds, genres, inv=False):
    """Process the results returned by a query and update the genres with the newly discovered genres and relations
    :param query: the SPARQL query to be executed
    :param sparql_dbp: DBPedia SPARQL endpoint
    :param relations: dictionary with genre relations that is updated
    :param seeds: the seeds of genres that still need to be crawled; it will be updated here too
    :param genres: the list of unique genres that is updated
    :param inv: a seed genre could be the source of an edge in the DBpedia graph or the target; depending on the role it plays the relations dictionary is updated differently
    """
    sparql_dbp.setQuery(query)
    results = sparql_dbp.query().convert()

    for result in results["results"]["bindings"]:
        prop = result["property"]["value"]
        rel = utils.rels_mapping[prop]
        if rel not in relations:
            relations[rel] = {}
        genre1 = result["genre1"]["value"]
        genre2 = result["genre2"]["value"]
        if not inv:
            if genre1 not in relations[rel]:
                relations[rel][genre1] = set()
            relations[rel][genre1].add(genre2)
        else:
            if genre2 not in relations[rel]:
                relations[rel][genre2] = set()
            relations[rel][genre2].add(genre1)

        #if 'wikiPageRedirects' not in prop and genre2 not in genres:
        if genre2 not in genres:
            seeds.append(genre2)


def collect_aliases_from_seeds(seeds, lang, genre_aliases):
    """Collect aliases from a list of music genre seeds
    :param seeds: the seed music genres
    :param lang: the targetted language
    :param genre_aliases: the dictionary to be updated
    """
    query_template = Template("""SELECT DISTINCT ?genre, ?alias
                  WHERE {
                    ?genre owl:sameAs ?alias.
                    FILTER (?genre IN ($list)).
                    FILTER ($other_lang_cond)
                  }""")
    endpoint = utils.get_endpoint_for_lang(lang)
    other_langs_cond = utils.get_alias_filter(lang, langs)
    sparql_dbpedia = SPARQLWrapper(endpoint + "sparql")
    sparql_dbpedia.setReturnFormat(JSON)

    start = 0
    while start < len(seeds):
        end = start + 50
        if end > len(seeds):
            end = len(seeds)
        #print(start, end)

        list_genres_str = utils.get_seeds_filter(seeds[start:end])
        start = end
        query = query_template.substitute({'list': list_genres_str, 'other_lang_cond': other_langs_cond})
        #print(query)
        sparql_dbpedia.setQuery(query)

        results = sparql_dbpedia.query().convert()
        for result in results["results"]["bindings"]:
            genre = result["genre"]["value"]
            alias = result["alias"]["value"]
            if genre not in genre_aliases:
                genre_aliases[genre] = set()
            genre_aliases[genre].add(alias)


def create_dbp_multigraph(genres, relations, genre_aliases):
    """Create the DBpedia music genre multidigraph
    :param genres: the list of discovered genres that will be graph nodes
    :param relations: the list of music genre relations that will be edges
    :param genre_aliases: the list of music genre aliases
    """
    G = nx.MultiGraph()
    for lang in genres:
        for g in genres[lang]:
            G.add_node(lang + ':' + g)
    for lang in relations:
        for r in relations[lang]:
            for g in relations[lang][r]:
                pref_g = lang + ':' + g
                for g2 in relations[lang][r][g]:
                    G.add_edge(pref_g, lang + ':' + g2, type=r)
    for lang in genre_aliases:
        for g in genre_aliases[lang]:
            pref_g = lang + ':' + g
            for g2 in genre_aliases[lang][g]:
                other_lang = utils.get_lang(g2)
                G.add_edge(pref_g, other_lang + ':' + g2, type='sameAs')
    return G


data_dir = utils.DATA_DIR

# First retrieve all entities of type MusicGenre from DBpedia for each language
print("Collect MusicGenre entities from DBpedia")
genres = {}
for lang in langs:
    genres[lang] = get_MusicGenres_ents(lang)
    print(lang, len(genres[lang]))

# Second collect alias genres for the previously discovered genres
print("Collect aliases for MusicGenre entities")
genre_aliases = {}
for lang in langs:
    genre_aliases[lang] = get_MusicGenres_aliases(lang, genres)
    print(lang, len(genres[lang]), len(genre_aliases[lang]))

# Add to genres those discovered through the music items from the corpus
print("Add the music genres from the corpus")
df = pd.read_csv(opj(data_dir, 'filtered_musical_items.csv'), index_col='id')
corpus_genres = utils.corpus_genres_per_lang(df)
for lang in corpus_genres:
    genres[lang].update(corpus_genres[lang])
    print(lang, len(genres[lang]))


# At this point, the high-confidence music genres have been collected
# either because they were marked as MusicGenre or they are aliases of
# MusicGenre entities or they were associated as genres to corpus items

# Now crawl relations starting from these genres
print("Crawl music genre relations for each genre")
relations = {}
for lang in langs:
    relations[lang] = collect_genres_from_seeds(genres[lang], lang)
    print(lang, len(genres[lang]))
    time.sleep(30)

# After finish crawling the relations,
# bring in the aliases of the newly discovered genres
print("Complete the aliases for all languages")
for lang in langs:
    seeds = genres[lang] - set(genre_aliases[lang].keys())
    print(lang, len(seeds))
    collect_aliases_from_seeds(list(seeds), lang, genre_aliases[lang])
    print(lang, len(genre_aliases[lang]))
    time.sleep(30)

# Create the graph
G = create_dbp_multigraph(genres, relations, genre_aliases)
nx.write_graphml(G, utils.RAW_GRAPH_PATH)
