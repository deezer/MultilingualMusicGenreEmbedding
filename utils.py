import re
import os
import ast
import pickle
from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import pandas as pd
import networkx as nx


# Relative path of the data dir
DATA_DIR = 'data/'
# Relative path of the acousticbrainz taxonomy info dir
ACOUSTICBRAINZ_DIR = 'data/acousticbrainz/'
# Relative path of the raw corpus
RAW_CORPUS_FILE_PATH = 'data/musical_items.csv'
# Relative path of the filtered corpus
CORPUS_FILE_PATH = 'data/filtered_musical_items.csv'
# Relative path of the filtered graph
RAW_GRAPH_PATH = 'data/dbp_multigraph.graphml'
# Relative path of the filtered graph
GRAPH_PATH = 'data/filtered_dbp_graph.graphml'
# Relative path of the folder containing the data folds
FOLDS_DIR = 'data/folds/'
# Relative path of the folder containing the data folds
TRIES_DIR = 'data/tries/'
# Relative path of the normalized multilingual genre graph
NORM_MULTILING_GRAPH_PATH = 'data/norm_multilang.graphml'
# Relative path of the normalized English only genre graph
NORM_EN_GRAPH_PATH = 'data/norm_acousticbrainz.graphml'
# Relative path of the aligned fastText embeddings
ALIGNED_FT_EMB_PATH = '/Users/eepure/Documents/Projects/MultiLangTagTranslation/data/aligned_embeddings/'

# Graph will be loaded only once and stored here
GRAPH = None
# Tags per language based on the graph
TAG_PER_LANG = None

# Languages supported
langs = ['en', 'fr', 'es']
# Strategies to compute multi-word expression embeddings from word embeddings
emb_composition_types = ['avg', 'wavg']
# Strategies in retrofitting; weighted is when edges are treated differently in the retrofitting method depending on their types (see paper)
retro_emb_types = ['weighted', 'unweighted']
# DBpedia types of music genre relations
rels_types = {'wikiPageRedirects', 'stylisticOrigin', 'musicSubgenre', 'musicFusionGenre', 'derivative', 'sameAs'}
# DBpedia types of music genre relations which signify equivalence
equiv_rels_types = {'wikiPageRedirects', 'sameAs'}

# DBpedia music genre types per language (their names are translated)
rels = {}
rels['en'] = ['http://dbpedia.org/ontology/wikiPageRedirects',
    'http://dbpedia.org/ontology/stylisticOrigin',
    'http://dbpedia.org/ontology/musicSubgenre',
    'http://dbpedia.org/ontology/derivative',
    'http://dbpedia.org/ontology/musicFusionGenre']
rels['fr'] = ['http://dbpedia.org/ontology/wikiPageRedirects',
    'http://fr.dbpedia.org/property/originesStylistiques',
    'http://fr.dbpedia.org/property/sousGenres',
    'http://fr.dbpedia.org/property/genresDérivés',
    'http://fr.dbpedia.org/property/genresAssociés']
rels['es'] = ['http://dbpedia.org/ontology/wikiPageRedirects',
    'http://es.dbpedia.org/property/origenMusical',
    'http://es.dbpedia.org/property/subgéneros',
    'http://es.dbpedia.org/property/derivados',
    'http://es.dbpedia.org/property/fusiones']

# Mapping DBpedia music genre relations to their English names
rels_mapping = {}
rels_mapping['http://dbpedia.org/ontology/wikiPageRedirects'] = 'wikiPageRedirects'
rels_mapping['http://dbpedia.org/ontology/stylisticOrigin'] = 'stylisticOrigin'
rels_mapping['http://dbpedia.org/ontology/musicSubgenre'] = 'musicSubgenre'
rels_mapping['http://dbpedia.org/ontology/derivative'] = 'derivative'
rels_mapping['http://dbpedia.org/ontology/musicFusionGenre'] = 'musicFusionGenre'
rels_mapping['http://fr.dbpedia.org/property/originesStylistiques'] = 'stylisticOrigin'
rels_mapping['http://fr.dbpedia.org/property/sousGenres'] = 'musicSubgenre'
rels_mapping['http://fr.dbpedia.org/property/genresDérivés'] = 'derivative'
rels_mapping['http://fr.dbpedia.org/property/genresAssociés'] = 'musicFusionGenre'
rels_mapping['http://es.dbpedia.org/property/origenMusical'] = 'stylisticOrigin'
rels_mapping['http://es.dbpedia.org/property/subgéneros'] = 'musicSubgenre'
rels_mapping['http://es.dbpedia.org/property/derivados'] = 'derivative'
rels_mapping['http://es.dbpedia.org/property/fusiones'] = 'musicFusionGenre'


def get_alias_filter(lang, langs):
    """ Format the part of query which retrieves aliases only in the languages of interest
    :param lang: the language of the DBpedia which is queried
    :param langs: the list of targetted languages
    :return: the formatted part of query which will be joined to the main query
    """
    other_langs_cond = ''
    for other_lang in langs:
        if other_lang == lang:
            continue
        if other_lang == 'en':
            other_langs_cond += 'strstarts(str(?alias), "http://dbpedia.org/") || '
        else:
            other_langs_cond += ''.join(['strstarts(str(?alias), "http://', other_lang, '.dbpedia.org/") || '])
    other_langs_cond = other_langs_cond[:-4]
    return other_langs_cond


def get_endpoint_for_lang(lang):
    """ Return the DBpedia endpoint for a specific language
    :param lang: the language of the DBpedia which is queried
    :return: the endpoint
    """
    if lang == 'en':
        endpoint = "http://dbpedia.org/"
    else:
        endpoint = "http://[LANG].dbpedia.org/".replace('[LANG]', lang)
    return endpoint


def get_lang(ent):
    """Extract the language from a DBpedia entity URL
    :param ent: the DBpedia entity url
    :return: the language code
    """
    if ent.startswith('http://dbpedia.org'):
        return 'en'
    tokens = re.findall(r'(?:https?://)(.{2})(?:\..+)', ent)
    if len(tokens) > 0:
        return tokens[0]
    return None


def get_seeds_filter(seeds):
    """Helper to format the part of the query which retrieves music genres for a seed list of music items provided by their URL
    :param seeds: seed music items
    :return: the formatted part of query which will be joined to the main query
    """
    list_genres_str = ''
    for g in seeds:
        if not g.startswith('http'):
            continue
        list_genres_str += ''.join(['<', g, '>, '])
    list_genres_str = list_genres_str[:-2]
    return list_genres_str


def corpus_genres_per_lang(df, min_count=1):
    """Get corpus genres per language which appear at least min_count times
    :param df: the corpus
    :param min_count: number of times a genre should appear, default 1
    :return: the genres per language that appear at least min_count times
    """
    selected_tags = {}
    for lang in langs:
        tags = []
        for annotations in df[lang].dropna().tolist():
            tags.extend(ast.literal_eval(str(annotations)))
        counter = Counter(tags)
        selected_tags[lang] = set()
        for x in counter:
            if counter[x] >= min_count:
                selected_tags[lang].add(x)
    return selected_tags


def get_genre_rels_filter(lang):
    """ Helper to format the part of the query which retrieves music genres by crawling genre relations
    :param lang: the language of the DBpedia which is queried
    :return: the formatted part of query which will be joined to the main query
    """
    cond = ''
    for i in range(len(rels[lang])):
        if i == len(rels[lang]) - 1:
            cond += ''.join(['<', rels[lang][i], '>'])
        else:
            cond += ''.join(['<', rels[lang][i], '>', ', '])
    return cond


def all_formatted_genres(df, norm_tags=True, as_set=True):
    """Get corpus music genre names
    :param df: the corpus dataframe
    :param norm_tags: specifies if tags are normalized or not
    :param as_set: specifies if the results is a dictionary with genres per language or a set containing all multilingual genres
    :return: the corpus music genre names
    """
    genres = corpus_genres_per_lang(df)
    all_genres = {}
    for lang in genres:
        all_genres[lang] = set()
        for g in genres[lang]:
            if norm_tags:
                g_name = get_ent_name(g)
            else:
                g_name = g
            all_genres[lang].add(g_name)
    if as_set:
        all_genres_set = set()
        for lang in all_genres:
            for g in all_genres[lang]:
                all_genres_set.add(''.join([lang + ':' + g]))
        return all_genres_set
    return all_genres


def get_ent_name(ent):
    """Extract the name of a DBpedia entity from its URL
    :param ent: the DBpedia URL of the entity
    :return: the entity name
    """
    tokens = re.findall(r"(?:\w{2}:)?(?:https?:\/\/\w{0,2}.?dbpedia.org\/resource\/)(.+(?!_)[\w\!])(?:$|(_?\(.+\)$))", ent)
    if len(tokens) == 0:
        return None
    return tokens[0][0]


def get_tags_for_source(source, graph_path=GRAPH_PATH):
    """Get unique music genres in the multilingual graph for a source
    :param source: the source / language
    :param graph_path: the graph file path
    :return: tags per source / language
    """
    global GRAPH
    global TAG_PER_LANG
    if TAG_PER_LANG is None or source not in TAG_PER_LANG:
        if GRAPH is None:
            GRAPH = nx.read_graphml(graph_path)
        TAG_PER_LANG = {}
        if source not in TAG_PER_LANG:
            for node in GRAPH:
                lang = node[:2]
                if lang not in TAG_PER_LANG:
                    TAG_PER_LANG[lang] = []
                TAG_PER_LANG[lang].append(node)
    return TAG_PER_LANG[source]


def get_graph(graph_path=GRAPH_PATH):
    """Returns the multilingual DBpedia-based music genre graph
    :param graph_path: the graph file path
    """
    global GRAPH
    if GRAPH is None:
        GRAPH = nx.read_graphml(graph_path)
    return GRAPH


def load_trie(lang):
    """Load trie for a language
    :param lang: the targetted language
    """
    lemma_trie = None
    with open(os.path.join(TRIES_DIR, lang, 'lemma_trie'), 'rb') as f:
        lemma_trie = pickle.load(f)
    return lemma_trie


def read_embeddings(path, sep=' '):
    """Read embeddings given in text format
    :param path: the embedding file path
    :param sep: the separator used in the file, default space
    :return: the embeddings as a dict and their dimension
    """
    embeddings = {}
    emb_dim = None
    with open(path, 'r', encoding='utf-8') as _:
        for line in _:
            values = line.rstrip().rsplit(sep)
            if len(values) == 2:
                emb_dim = int(values[1])
            else:
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings[word] = coefs
    return embeddings, emb_dim









def load_tag_csv(path, sources=langs, sep='\t'):
    df = pd.read_csv(path, sep=sep)

    def load_row(r):
        if isinstance(r, float):
            return []
        else:
            return eval(r)

    def format_values(r):
        formatted_r = []
        for v in r:
            formatted_r.append(get_ent_name(v))
        return formatted_r

    for source in sources:
        df[source] = df[source].apply(load_row)
        df[source] = df[source].apply(format_values)
    return df


def format_tags_for_source(tags, source):
    return [source + ":" + t for t in tags]


def format_dataset_rows_and_split(df, sources, target):
    train_data = []
    target_data = []
    for t in zip(*[df[s] for s in list(sources) + [target]]):
        stags = t[:len(sources)]
        ttags = t[-1]
        train_data.append([])
        for i, s in enumerate(sources):
            train_data[-1].extend(format_tags_for_source(stags[i], s))
        target_data.append(format_tags_for_source(ttags, target))
    return train_data, target_data


class TagManager:
    _MANAGERS_ = {}

    def __init__(self, sources, target):
        self._sources = sources
        self._target = target
        self.source_tags = [el for source in sources for el in get_tags_for_source(source)]
        self.target_tags = [el for el in get_tags_for_source(target)]
        self._mlb_sources = None
        self._mlb_target = None

    @property
    def sources(self):
        return self._sources

    @property
    def target(self):
        return self._target

    @property
    def mlb_sources(self):
        if self._mlb_sources is None:
            self._mlb_sources = MultiLabelBinarizer(classes=self.source_tags, sparse_output=True)
            self._mlb_sources.fit([[]])
        return self._mlb_sources

    @property
    def mlb_target(self):
        if self._mlb_target is None:
            self._mlb_target = MultiLabelBinarizer(classes=self.target_tags, sparse_output=True)
            self._mlb_target.fit([[]])
        return self._mlb_target

    def transform_for_target(self, df, as_array=False):
        if as_array:
            return self.mlb_target.transform(df).toarray().astype("float32")
        else:
            return self.mlb_target.transform(df)

    def transform_for_sources(self, df, as_array=False):
        if as_array:
            return self.mlb_sources.transform(df).toarray().astype("float32")
        else:
            return self.mlb_sources.transform(df)

    @staticmethod
    def normalize_tag(tag, prefixed=True, asList=False):
        if prefixed:
            tag = tag[3:]
        return TagManager._norm_basic(tag, asList=asList)

    @staticmethod
    def normalize_tag_wtokenization(tag, trie, prefixed=True, asList=False):
        """Normalize a tag and then apply a trie split
        """
        words = set()
        tag_tokens = TagManager.normalize_tag(tag, prefixed, asList=True)
        for t in tag_tokens:
            if len(t) <= 3 or trie.has_word(t):
                words.add(t)
            else:
                tokens = trie.tokenize(t)
                if len(tokens) == 0:
                    words.add(t)
                else:
                    words.update(tokens)
            if '' in words:
                words.remove('')
        sorted_words = sorted(words)
        if asList:
            return sorted_words
        else:
            return ' '.join(sorted_words)

    @staticmethod
    def normalize_tag_basic(t, prefixed=True):
        if prefixed:
            t = t[3:]
        return t.lower().replace('_', '-').replace(',', '')

    @staticmethod
    def _norm_basic(s, asList=False, sort=False):
        """
        Perform a basic normalization
        -lower case
        -replace special characters by space
        -sort the obtained words
        """
        split_chars_gr1 = ['_', '-', '/', ',', '・']
        split_chars_gr2 = ['(', ')', "'", "’", ':', '.', '!', '‘', '$']
        s = list(s.lower())
        new_s = []
        for c in s:
            if c in split_chars_gr1:
                new_s.append(' ')
            elif c in split_chars_gr2:
                continue
            else:
                new_s.append(c)
        new_s = ''.join(new_s)
        if sorted or asList:
            words = new_s.split()
            if sort:
                words = sorted(words)
                return ' '.join(words)
            if asList:
                return words
        return new_s

    @classmethod
    def get(cls, sources, target):
        sources_key = " ".join(sorted(sources))
        if sources_key not in cls._MANAGERS_ or target not in cls._MANAGERS_[sources_key]:
            m = TagManager(sources, target)
            cls._MANAGERS_.setdefault(sources_key, {})
            cls._MANAGERS_[sources_key][target] = m
        return cls._MANAGERS_[sources_key][target]


class DataHelper:

    def __init__(self, tag_manager, dataset_path=None):
        self.tag_manager = tag_manager
        self.dataset_path = dataset_path
        self.dataset_df = None
        if self.dataset_path is not None:
            self._load_dataset()

    def _load_dataset(self):
        print("Loading dataset...")
        self.dataset_df = load_tag_csv(self.dataset_path)
        print("Loaded.")

    def get_test_data(self, fold, as_array=True):
        return self._get_dataset_split(fold, as_array)

    def _get_dataset_split(self, fold, as_array):
        bool_index = self.dataset_df.fold == fold
        df = self.dataset_df[bool_index]
        train_data, target_data = format_dataset_rows_and_split(df, self.tag_manager.sources, self.tag_manager.target)
        return self.transform_sources_and_target_data(train_data, target_data, as_array)

    def transform_sources_and_target_data(self, source_df, target_df, as_array):
        return self.tag_manager.transform_for_sources(source_df, as_array), self.tag_manager.transform_for_target(target_df, as_array)
