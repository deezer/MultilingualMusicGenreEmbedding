import numpy as np
import networkx as nx

from utils import read_translation_table, load_trie
from tag_translation.base_translator import Translator


class DbpMappingTranslator(Translator):
    """ Baseline translator used in the English-language experiment with the AcousticBrainz data
    The prediction relies on a pre-computed translation table
    For more details about the computation of the translation table, please check our previous work: Epure, E. V., Khlif, A., & Hennequin, R. (2019). Leveraging Knowledge Bases And Parallel Annotations For Music Genre Translation. ISMIR 2019.
    """
    def __init__(self, tag_manager, table_path):
        """Constructor requires a tag_manager and the path to where the translation table is stored
        :param tag_manager: object of type TagManager (in tag_manager.py)
        :param table_path: the filepath to the translation table
        """
        self.tag_manager = tag_manager
        self.source_genres = self.tag_manager.source_tags
        self.target_genres = self.tag_manager.target_tags

        self.norm_table = read_translation_table(table_path, tag_manager)
        self.W = self.norm_table.values.T
        self.b = np.zeros((1,))


class GraphDistanceMapper(Translator):
    """ Baseline translator used in the multilingual experiment
        It computes distances between tags based on the multilingual DBpedia-based multilingual music genre graph
    """

    def __init__(self, tag_manager, G, langs):
        """Constructor
        :param tag_manager: object of type TagManager (in tag_manager.py)
        :param G: the multilingual DBpedia-based multilingual music genre graph
        :param langs: the languages to be considered
        """
        self.tag_manager = tag_manager
        self.tries = {}
        for lang in langs:
            self.tries[lang] = load_trie(lang)
        source_tags = self.get_source_tags()
        target_tags = self.get_target_tags()

        tbl = np.zeros((len(source_tags), len(target_tags)))
        # Cutoff is set 2 because in this way we can retrieve the direct
        # translation and the neighbours of that translation
        spaths = dict(nx.all_pairs_shortest_path_length(G, cutoff=2))
        for i in range(len(source_tags)):
            sg = source_tags[i]
            for j in range(len(target_tags)):
                tg = target_tags[j]
                if sg in spaths and tg in spaths[sg]:
                    d = -spaths[sg][tg]
                else:
                    d = -len(G)
                tbl[i, j] = d
        self.W = tbl.T

    def get_source_tags(self):
        """Return the list of formatted and normalized source tags
        """
        return self._get_norm_tags(self.tag_manager.source_tags)

    def get_target_tags(self):
        """Return the list of formatted and normalized target tags
        """
        return self._get_norm_tags(self.tag_manager.target_tags)

    def _get_norm_tags(self, tags):
        """Return normalized list of tags
        :param tags: input tags
        """
        norm_tags = []
        for tag in tags:
            lang = tag[0:2]
            norm_tags.append(lang + ':' + self.tag_manager.normalize_tag_wtokenization(tag, self.tries[lang]))
        return norm_tags
