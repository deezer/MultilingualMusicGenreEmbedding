import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.utils import read_translation_table, load_trie
from tag_translation.base_translator import Translator


class EmbsTranslator(Translator):
    """Translator based on pre-computed embeddings
    """

    def __init__(self, tag_manager, embs_path, langs):
        """Constructor
        :param tag_manager: an instance of class type TagManager
        :param embs_path: path to the embeddings
        :param langs: the languages handled, used to load the tries
        """
        self.tag_manager = tag_manager
        self.tries = {}
        for lang in langs:
            self.tries[lang] = load_trie(lang)
        norm_sts = self.get_source_tags()
        norm_tts = self.get_target_tags()

        model = read_translation_table(embs_path).T
        known_genres = set(model.columns)
        tt = pd.DataFrame(cosine_similarity(model.T),index=model.columns, columns=model.columns)

        ns = len(tag_manager.source_tags)
        nt = len(tag_manager.target_tags)
        d = np.zeros((ns, nt))
        for i in range(ns):
            norm_s = norm_sts[i]
            if norm_s not in known_genres:
                continue
            for j in range(nt):
                norm_t = norm_tts[j]
                if norm_t not in known_genres:
                    continue
                d[i, j] = tt.at[norm_s, norm_t]

        self.W = pd.DataFrame(d.T, index=tag_manager.target_tags, columns=tag_manager.source_tags)
        print(self.W)

    def get_source_tags(self):
        """Return a list of source tags, the format could vary between the 2 experiments English-language only and multilingual
        """
        raise NotImplementedError("")

    def get_target_tags(self):
        """Return a list of target tags, the format could vary between the 2 experiments English-language only and multilingual
        """
        raise NotImplementedError("")


class EnglishLangEmbsTranslator(EmbsTranslator):
    """Translator used in the English-language only experiment
    """
    def get_source_tags(self):
        """Return the list of formatted and normalized source tags
        """
        return ['en:' + self.tag_manager.normalize_tag_wtokenization(t, self.tries['en'], prefixed=False) for t in self.tag_manager.unprefixed_source_tags]

    def get_target_tags(self):
        """Return the list of formatted and normalized target tags
        """
        return ['en:' + self.tag_manager.normalize_tag_wtokenization(t, self.tries['en'], prefixed=False) for t in self.tag_manager.unprefixed_target_tags]


class MultilingualEmbsTranslator(EmbsTranslator):
    """Translator used in the multilingual experiment
    """

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
