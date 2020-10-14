from sklearn.preprocessing import MultiLabelBinarizer

import utils


class TagManager:
    """ Tag Manager class used for normalizing tags and preparing multilabelbinarizer objects used in evaluation"""
    _MANAGERS_ = {}

    def __init__(self, sources, target, source_tags, target_tags):
        self._sources = sources
        self._target = target
        self.source_tags = source_tags
        self.target_tags = target_tags
        self._mlb_sources = None
        self._mlb_target = None

    @property
    def sources(self):
        return self._sources

    @property
    def target(self):
        return self._target

    @property
    def unprefixed_source_tags(self):
        return [t.split(":")[1] for t in self.source_tags]

    @property
    def unprefixed_target_tags(self):
        return [t.split(":")[1] for t in self.target_tags]

    @property
    def mlb_sources(self):
        """ Create a MultiLabelBinarizer from the source tags"""
        if self._mlb_sources is None:
            self._mlb_sources = MultiLabelBinarizer(classes=self.source_tags, sparse_output=True)
            self._mlb_sources.fit([[]])
        return self._mlb_sources

    @property
    def mlb_target(self):
        """ Create a MultiLabelBinarizer from the target tags"""
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
        """ Normalize a tag
        :param tag: input tag
        :param prefixed: if the tag is prefixed with the language code
        :param asList: if the tag tokens are returned as list of concatenated
        :return: the normalized tag
        """
        if prefixed:
            tag = tag[3:]
        return TagManager._norm_basic(tag, asList=asList)

    @staticmethod
    def normalize_tag_wtokenization(tag, trie, prefixed=True, asList=False):
        """Normalize a tag and then apply a trie split
        :param tag: input tag
        :param trie: used for attempting to split the tag in multiple tokens
        :param prefixed: if the tag is prefixed with the language code
        :param asList: if the tag tokens are returned as list of concatenated
        :return: the normalized tag which was also tokenized with the trie
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
    def _norm_basic(s, asList=False, sort=False):
        """Perform a basic normalization
        -lower case
        -replace special characters by space
        -sort the obtained words
        :param s: the input string / tag
        :param asList: if the tag tokens are returned as list of concatenated
        :param sort: if the obtained tokens are sorted before concatenation
        :return: the normalized tag
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
    def get(cls, sources, target, source_tags, target_tags):
        """ Returns a instance of a tag manager for the specific sources and target"""
        sources_key = " ".join(sorted(sources))
        if sources_key not in cls._MANAGERS_ or target not in cls._MANAGERS_[sources_key]:
            m = TagManager(sources, target, source_tags, target_tags)
            cls._MANAGERS_.setdefault(sources_key, {})
            cls._MANAGERS_[sources_key][target] = m
        return cls._MANAGERS_[sources_key][target]
