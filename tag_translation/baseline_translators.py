import sys
import numpy as np
import pandas as pd
import pickle
from nltk.metrics import edit_distance
import networkx as nx

from tag_translation.translator import Translator
import trie

from sklearn.metrics.pairwise import cosine_similarity


class BaselineTranslator(Translator):
    def __init__(self, tag_manager):
        self.tag_manager = tag_manager
        self.W = self.compute_distances()

    def get_distance(self, str1, str2):
        raise NotImplementedError("")

    def get_source_tags(self):
        raise NotImplementedError("")

    def get_target_tags(self):
        raise NotImplementedError("")

    def compute_distances(self):
        s_genres = self.get_source_tags()
        t_genres = self.get_target_tags()
        ns = len(s_genres)
        nt = len(t_genres)
        d = np.zeros((ns, nt))
        for i in range(ns):
            for j in range(nt):
                #print(s_genres[i], t_genres[j])
                d[i, j] = self.get_distance(s_genres[i], t_genres[j])
        return d.T

    def predict_scores(self, eval_data):
        norm = np.count_nonzero(eval_data, axis=1)
        print(norm)
        return eval_data.dot(self.W.T) / norm.reshape(norm.shape[0], 1)


class LevenshteinTranslator(BaselineTranslator):

    def get_distance(self, str1, str2):
        return 1 - edit_distance(str1, str2) / max(len(str1), len(str2))

    def get_source_tags(self):
        return self.tag_manager.unprefixed_source_tags

    def get_target_tags(self):
        return self.tag_manager.unprefixed_target_tags


class NormDirectTranslator(BaselineTranslator):

    def __init__(self, tag_manager, input_dir):
        #self.tag_rep, _, _, _ = load_genre_representation(input_dir)

        sys.modules['trie'] = trie
        with open('/Users/eepure/Documents/Projects/MultiLangTagTranslation/ismir_data/tries/en/en_words_trie', 'rb') as f:
            self.lang_trie = pickle.load(f)
            print('dict trie loaded')

        with open('/Users/eepure/Documents/Projects/MultiLangTagTranslation/ismir_data/tries/en/lemma_trie', 'rb') as f:
            self.lemma_trie = pickle.load(f)
            print('lemma trie loaded')

        super().__init__(tag_manager)

    def get_distance(self, str1, str2):
        return int(str1 == str2)

    def norm_basic(self, s, asList=False):
        split_chars = ['_', '-', '(', ')', '/', '\\', ',', "'", "’", ':', ';', '.', '!', '?', '‘', '&', ' ']
        s = list(s.lower())
        words = set()
        first_ind = 0
        for i in range(len(s)):
            if s[i] in split_chars:
                words.add(''.join(s[first_ind:i]))
                first_ind = i + 1
        # add the last word
        if first_ind < len(s):
            words.add(''.join(s[first_ind:len(s)]))

        sorted_words = sorted(words)
        if asList:
            return sorted_words
        return ' '.join(sorted_words).strip()

    def decode_genre(self, genre, asList=False):
        words = set()
        genre_tokens = self.norm_basic(genre, asList=True)
        for g in genre_tokens:
            if len(g) <= 3 or self.lemma_trie.has_word(g):
                words.add(g)
            else:
                tokens = self.lemma_trie.tokenize(g)
                if len(tokens) == 0:
                    words.add(g)
                else:
                    words.update(tokens)
            if '' in words:
                words.remove('')

        sorted_words = sorted(words)
        if asList:
            return sorted_words
        else:
            return " ".join(sorted_words)

    def get_source_tags(self):
        return [self.decode_genre(t) for t in self.tag_manager.unprefixed_source_tags]

    def get_target_tags(self):
        return [self.decode_genre(t) for t in self.tag_manager.unprefixed_target_tags]


class WordEmbTranslator(NormDirectTranslator):

    def __init__(self, tag_manager, input_dir, model):
        self.model = model
        super().__init__(tag_manager, input_dir)
        print(self.W)

    def get_distance(self, str1, str2):
        emb1 = self._get_mean_emb(str1)
        emb2 = self._get_mean_emb(str2)
        return cosine_similarity(emb1, emb2)

    def _get_mean_emb(self, tag):
        emb = np.zeros(self.model.wv.vectors.shape[1])
        n = 0
        for w in tag.split(' '):
            if w in self.model.wv.vocab:
                emb += self.model[w]
                n += 1
            #else:
            #    print(w)
        if n > 0:
            emb /= n
        return emb.reshape((1, len(emb)))

class WordEmbTranslator2(NormDirectTranslator):

    def __init__(self, tag_manager, input_dir, model):
        self.model = model
        super().__init__(tag_manager, input_dir)
        print(self.W)

    def get_distance(self, str1, str2):
        #print(str1, str2)
        str1 = 'en:' + str1
        str2 = 'en:' + str2
        if str1 not in self.model or str2 not in self.model:
            return 0

        a1 = self.model[str1].values
        a2 = self.model[str2].values
        return cosine_similarity(a1.reshape(1, -1), a2.reshape(1, -1))
        #return np.dot(a1, a2)


class WordEmbTranslator3(Translator):
    def __init__(self, tag_manager, model):
        sys.modules['trie'] = trie
        self.lemma_tries = {}
        for lang in ['en', 'fr', 'es']:
            with open('/Users/eepure/Documents/Projects/MultiLangTagTranslation/ismir_data/tries/' + lang + '/lemma_trie', 'rb') as f:
                self.lemma_tries[lang] = pickle.load(f)
                print('lemma trie loaded for ' + lang)

        self.tag_manager = tag_manager

        ns = len(tag_manager.source_tags)
        nt = len(tag_manager.target_tags)
        known_genres = set(model.columns)
        norm_sts, norm_tts = self.get_norm_genres()
        tt = pd.DataFrame(cosine_similarity(model.T),index=model.columns, columns=model.columns)

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

    def get_norm_genres(self):
        norm_sts = [self.decode_genre(g) for g in self.tag_manager.source_tags]
        norm_tts = [self.decode_genre(g) for g in self.tag_manager.target_tags]
        return norm_sts, norm_tts

    def decode_genre(self, tag, asList=False):
        words = set()
        i = tag.index(':')
        lang = tag[0:i]
        genre = tag[i + 1:]
        genre_tokens = self.norm_basic(genre, asList=True)
        for g in genre_tokens:
            if len(g) <= 3 or self.lemma_tries[lang].has_word(g):
                words.add(g)
            else:
                tokens = self.lemma_tries[lang].tokenize(g)
                if len(tokens) == 0:
                    words.add(g)
                else:
                    words.update(tokens)
            if '' in words:
                words.remove('')

        sorted_words = sorted(words)
        if asList:
            return sorted_words
        else:
            return lang + ":" + " ".join(sorted_words)

    def norm_basic(self, s, asList=False):
        split_chars = ['_', '-', '(', ')', '/', '\\', ',', "'", "’", ':', ';', '.', '!', '?', '‘', '&', ' ']
        s = list(s.lower())
        words = set()
        first_ind = 0
        for i in range(len(s)):
            if s[i] in split_chars:
                words.add(''.join(s[first_ind:i]))
                first_ind = i + 1
        # add the last word
        if first_ind < len(s):
            words.add(''.join(s[first_ind:len(s)]))

        sorted_words = sorted(words)
        if asList:
            return sorted_words
        return ' '.join(sorted_words).strip()

    def predict_scores(self, eval_data):
        norm = np.count_nonzero(eval_data, axis=1)
        print(norm)
        return eval_data.dot(self.W.T) / norm.reshape(norm.shape[0], 1)


class GraphDistanceMapper(Translator):
    def __init__(self, tag_manager, G):
        self.tag_manager = tag_manager
        sys.modules['trie'] = trie
        self.lemma_tries = {}
        for lang in ['en', 'fr', 'es']:
            with open('/Users/eepure/Documents/Projects/MultiLangTagTranslation/ismir_data/tries/' + lang + '/lemma_trie', 'rb') as f:
                self.lemma_tries[lang] = pickle.load(f)
                print('lemma trie loaded for ' + lang)

        source_tags, target_tags = self.get_norm_genres()

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

    def predict_scores(self, eval_data):
        norm = np.count_nonzero(eval_data, axis=1)
        print(norm)
        return eval_data.dot(self.W.T) / norm.reshape(norm.shape[0], 1)

    def get_norm_genres(self):
        norm_sts = [self.decode_genre(g) for g in self.tag_manager.source_tags]
        norm_tts = [self.decode_genre(g) for g in self.tag_manager.target_tags]
        return norm_sts, norm_tts

    def decode_genre(self, tag, asList=False):
        words = set()
        i = tag.index(':')
        lang = tag[0:i]
        genre = tag[i + 1:]
        genre_tokens = self.norm_basic(genre, asList=True)
        for g in genre_tokens:
            if len(g) <= 3 or self.lemma_tries[lang].has_word(g):
                words.add(g)
            else:
                tokens = self.lemma_tries[lang].tokenize(g)
                if len(tokens) == 0:
                    words.add(g)
                else:
                    words.update(tokens)
            if '' in words:
                words.remove('')

        sorted_words = sorted(words)
        if asList:
            return sorted_words
        else:
            return lang + ":" + " ".join(sorted_words)

    def norm_basic(self, s, asList=False):
        split_chars = ['_', '-', '(', ')', '/', '\\', ',', "'", "’", ':', ';', '.', '!', '?', '‘', '&', ' ']
        s = list(s.lower())
        words = set()
        first_ind = 0
        for i in range(len(s)):
            if s[i] in split_chars:
                words.add(''.join(s[first_ind:i]))
                first_ind = i + 1
        # add the last word
        if first_ind < len(s):
            words.add(''.join(s[first_ind:len(s)]))

        sorted_words = sorted(words)
        if asList:
            return sorted_words
        return ' '.join(sorted_words).strip()

