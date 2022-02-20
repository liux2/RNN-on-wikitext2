class Vocab(object):
    """ Converts word tokens to indices, and vice versa. """

    def __init__(self, freqs, corpus, window_size):
        super().__init__()
        self.indix2token = tuple(freqs)
        self.token2index = {k: v for v, k in enumerate(self.indix2token)}
        self.corpus = corpus
        self.window_size = window_size
        self.encoded_list = []
        self.data, self.target = self.encoding()

    def __len__(self):
        return len(self.encoded_list)

    def __getitem__(self, key):
        return torch.tensor(self.data[key]), torch.tensor(self.target[key])

    def encoding(self):
        def retrive(key):
            if isinstance(key, int):
                return None
            else:
                return self.token2index[key]

        encoded_list = [retrive(i) for i in self.corpus]
        self.encoded_list = [
            encoded_list[i : i + self.window_size]
            for i in range(0, len(encoded_list), self.window_size)
            if len(encoded_list[i : i + self.window_size]) == self.window_size
        ]
        data = [s[:-1] for s in self.encoded_list]
        target = [s[1:] for s in self.encoded_list]
        return data, target

    def decoding(self):
        def retrive(self, key):
            if isinstance(key, int):
                return self.indix2token[key]
            else:
                return None

        decoded_list = [[retrive(w) for w in s] for s in self.corpus]
