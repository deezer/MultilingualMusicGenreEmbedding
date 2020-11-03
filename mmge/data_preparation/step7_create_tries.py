import os
import spacy
import pickle


from mmge.utils import utils
from mmge.utils.trie import Trie
from mmge.utils.tag_manager import TagManager


opj = os.path.join


def create_trie(strings, word_min_len=1):
    """Create a trie from a list of strings
    :param strings: the input strings
    :param word_min_len: threshold specifying the minimum word length to be added to the trie
    :return: the resulting trie object
    """
    trie = Trie()
    for s in strings:
        if len(s) >= word_min_len:
            trie.add(s)
    return trie


# Use spacy for lemmatization
print("Loading spacy")
nlp = {}
for lang in utils.langs:
    if lang == 'en':
        nlp[lang] = spacy.load(lang)
    else:
        nlp[lang] = spacy.load(lang + "_core_news_sm")

# Normalize all genre words from the multilingual graph
print("Normalize music genre tags and obtain the unique words")
# a dict of normalized genre words per language
norm_genre_words = {}
for lang in utils.langs:
    norm_genre_words[lang] = set()
    tags = utils.get_tags_for_source(lang)
    for tag in tags:
        norm_genre_words[lang].update(TagManager.normalize_tag(tag, prefixed=True, asList=True))
print(norm_genre_words)

# Create trie per language
print("Create the tries for all languages")
for lang in utils.langs:
    lang_out_dir = opj(utils.TRIES_DIR, lang)
    if not os.path.exists(lang_out_dir):
        os.makedirs(lang_out_dir)

    # From the lemmatized music genre words
    lemmas = set()
    for word in norm_genre_words[lang]:
        # If the word is short do not lemmatize
        if len(word) <= 2:
            lemmas.add(word)
            continue
        doc = nlp[lang](word)
        for t in doc:
            lemmas.add(t.lemma_)
    lemma_trie = create_trie(lemmas, word_min_len=1)
    with open(opj(lang_out_dir, "lemma_trie"), 'wb') as _:
        pickle.dump(lemma_trie, _)

