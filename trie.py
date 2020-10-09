class TrieNode(object):
    """
    Trie Node
    """

    def __init__(self, char):
        # The character
        self._char = char

        # The children nodes,keys are chars, values are nodes
        self._children = {}

        # Marks if the node is the last character of the word
        self._is_word = False

        # Keeps track of the actual word
        self._word = None

    @property
    def isWord(self):
        return self._is_word

    @isWord.setter
    def isWord(self, value):
        self._is_word = value


class Trie(object):
    """
        Trie implementation, adjusted for genre representation
    """

    def __init__(self):
        self._root = TrieNode("*")

    def tokenize(self, string):
        """
            Given a string tokenize it into words that are present
            in the Trie. If not successful, returns an empty string ''
        """
        def _tokenize(i, node=None, tk=[]):

            if node is None:
                node = self._root

            if i >= len(string):
                if node.isWord:
                    return tk
                else:
                    return []
            else:
                c = string[i]
                if c in node._children:
                    if node.isWord:
                        # greedily trying to match the longest word possible
                        tk1 = _tokenize(i + 1, node=node._children[c], tk=[c])
                        if tk1:
                            # if that recursive branch succeeds return result
                            tk.extend(tk1)
                            return tk
                        else:
                            # the above recursion failed; be less greedy
                            # take the next best match path
                            tk.append(" ")
                            return _tokenize(i, node=None, tk=tk)
                    else:
                        tk.append(c)
                        return _tokenize(i + 1, node=node._children[c], tk=tk)
                else:
                    if node.isWord:
                        tk.append(" ")
                        return _tokenize(i, node=None, tk=tk)
                    else:
                        return []

        # Result is a list of chars, with space to delimit each word
        tk = _tokenize(i=0)
        if len(tk) == 0:
            tokens = []
        else:
            tokens = ''.join(tk).split(' ')
        return tokens

    def _has_prefix(self, prefix, check_has_word):
        """
            Check if a prefix exists in the trie
            if check_has_word is True, then it checks whether the word to exist
        """
        if prefix is None:
            return False

        chars = list(prefix)
        node = self._root
        for char in chars:
            if char not in node._children:
                return False
            node = node._children[char]

        if check_has_word:
            return node.isWord
        else:
            return True

    def longest_prefix(self, prefix):
        """
            Check if a prefix exists in the trie
            if check_has_word is True, then it checks whether the word to exist
        """
        if prefix is None:
            return None
        chars = list(prefix)
        node = self._root
        for i in range(len(chars)):
            char = chars[i]
            if char not in node._children:
                return ''.join(chars[:i])
            node = node._children[char]
        return None

    def has_prefix(self, prefix):
        """
            Check if a prefix exists in the trie
        """
        return self._has_prefix(prefix, False)

    def has_word(self, prefix):
        """
            Check if a word exists in the trie
        """
        return self._has_prefix(prefix, True)

    def add(self, word):
        """
            Add a word in the trie
        """
        chars = list(word)
        node = self._root

        for char in chars:
            if char not in node._children:
                node._children[char] = TrieNode(char)
            node = node._children[char]

        node.isWord = True
        node._word = word

    def get_words(self, prefix=None):
        """
            Return all the words in the trie, having the given prefix
            If prefix is None, it returns all the words
        """
        def _get_word_list(node, prefix, result):
            if node.isWord:
                result.append(prefix)
            for char in node._children.keys():
                _get_word_list(node._children[char], prefix + char, result)

        if prefix is None:
            return None

        result = []
        node = self._root
        chars = list(prefix)

        for char in chars:
            if char not in node._children:
                return result
            node = node._children[char]
        _get_word_list(node, prefix, result)
        return result

    def get_all_words(self):
        """
            Return all the words in the trie
        """
        return self.get_words("")

    def print_words(self, prefix=""):
        """
            Print all words that start with given prefix
            If prefix is "", all the words of the trie are printed
        """
        result = self.get_words(prefix)
        for word in result:
            print(word)
