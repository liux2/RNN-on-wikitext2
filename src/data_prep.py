from nltk.tokenize import word_tokenize, sent_tokenize
import random


class Prep:
    """Preparing tokenization and frequences."""

    def __init__(self):
        with open("./data/wiki.test.txt") as f:
            self.test = f.read()
        with open("./data/wiki.train.txt") as f:
            self.train = f.read()
        with open("./data/wiki.valid.txt") as f:
            self.valid = f.read()
        # self.test1 = "After release , it received downloadable content . along with an expanded edition in November of that year ."
        # self.test2 = "After it received ."
        self.word_freqs = {"<oov>": 1}

    def tokenize(self, corpus):
        """
        Tokenized the lines, remove the titles, and make it lowercase,
        return lines list.
        list[list[word]]
        """

        # Create token list
        sent_tokens = [word_tokenize(t) for t in sent_tokenize(corpus)]
        random.shuffle(sent_tokens)
        word_tokens = [[w.lower() for w in s] for s in sent_tokens]

        # Remove last punctuation, add <s></s>
        word_tokens = [
            ["<s>"] + s + ["</s>"] if s[-1].isalnum() else ["<s>"] + s[:-1] + ["</s>"]
            for s in word_tokens
        ]
        corpus = []
        for s in word_tokens:
            corpus.extend(s)
        return corpus

    def building_vocab(self, corpus):
        """Building vocab list from training set."""
        for w in corpus:
            # the word has already been found
            if w in self.word_freqs:
                self.word_freqs[w] += 1
            # the word has not yet already been found
            else:
                self.word_freqs[w] = 1
