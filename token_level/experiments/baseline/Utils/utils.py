from collections import defaultdict
import torch

class SetVocab(dict):
    def __init__(self, vocab):
        self.update(vocab)

    def ws2ids(self, ws):
        return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if i in idx2w else "UNK" for i in ids]

class Vocab(defaultdict):
    def __init__(self, train=True):
        super().__init__(lambda : len(self))
        self.train = train
        self.UNK = "UNK"
        self.pad = "<pad>"
        # set UNK token to 0 index
        self[self.UNK]
        self[self.pad]

    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return [self[w] for w in ws]
        else:
            return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if int(i) in idx2w else "UNK" for i in ids]


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
