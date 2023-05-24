import torch
import torch.nn as nn
import json
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import argparse

from Utils.WordVecs import WordVecs
from Utils.utils import Vocab, prepare_sequence

from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score


def convert_gold(golds, labels2idx):
    new_gold = []
    for gold in golds:
        g = []
        for labels in gold:
            t = torch.zeros(len(labels2idx))
            for label in labels:
                t[labels2idx[label]] = 1.0
            g.append(t)
        new_gold.append(torch.stack(g))
    return new_gold

class MulticlassBert(nn.Module):
    def __init__(self,
                 label2idx,
                 #model_name="bert-base-multilingual-cased",
                 model_name="NbAiLab/nb-bert-base"
                 ):
        super(MulticlassBert, self).__init__()
        self.model_name = model_name
        self.tagset_size = len(label2idx)
        self.label2idx = label2idx
        self.idx2label = dict([(v, k) for k, v in label2idx.items()])
        self.model = AutoModel.from_pretrained(model_name, from_tf=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.linear = nn.Linear(self.model.config.hidden_size, self.tagset_size)
        self.loss = nn.BCEWithLogitsLoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-5)

    def shuffle(self, x, y):
        i = np.arange(len(x))
        np.random.shuffle(i)
        x, y = (np.array(x, dtype=object)[i], np.array(y, dtype=object)[i])
        return list(x), list(y)

    def batch(self, x, y, batch_size=5, shuffle=False):
        batches = []
        if shuffle:
            x, y = self.shuffle(x, y)
        idx = 0
        for batch in range(len(x) // batch_size):
            x_i = x[idx:idx+batch_size]
            y_i = y[idx:idx+batch_size]
            batches.append((x_i, y_i))
            idx += batch_size
        if len(x) % batch_size > 0:
            x_i = x[idx:]
            y_i = y[idx:]
            batches.append((x_i, y_i))
        return batches
    #
    def forward(self, sentences):
        encodings = []
        for sent in sentences:
            sent_reps = []
            # create an averaged representation for each token
            for token in sent:
                tokenized = self.tokenizer(token, return_tensors='pt', add_special_tokens=False)
                subtokens = self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
                tok_reps_cl_emb = self.model(**tokenized)
                # average all subtoken embeddings
                token_reps = tok_reps_cl_emb[0].squeeze(0).mean(0)
                sent_reps.append(token_reps)
            # check that the number of tokens and number of embeddings is equal
            assert len(sent) == len(sent_reps)
            sent_pred = self.linear(torch.stack(sent_reps))
            encodings.append(sent_pred)
        return encodings
    #
    def get_loss(self, sentences, golds):
        out = self.forward(sentences)
        pred = torch.cat(out)
        gold = torch.cat(golds)
        return self.loss(pred, gold)
    #
    def train_model(self, train_x, train_y, dev, epochs=10, batch_size=100):
        for epoch in range(epochs):
            print(f"epoch {epoch + 1}")
            batches = self.batch(train_x, train_y, batch_size, shuffle=True)
            epoch_loss = 0
            self.train()
            for sentences, golds in tqdm(batches):
                self.zero_grad()
                loss = self.get_loss(sentences, golds)
                loss.backward()
                epoch_loss += loss.data
                self.optim.step()
            print(epoch_loss)
            acc, f1 = self.test(dev)
            print(f"dev acc: {acc:.3f}")
            print(f"dev f1: {f1:.3f}")

    def predict(self, sents, cutoff=0.5, batch_size=10):
        pred_labels = []
        self.eval()
        out = self.forward(sents)
        for pred_sent in out:
            sent_labels = []
            pred_sent = torch.sigmoid(pred_sent)
            preds = pred_sent > cutoff
            for pred_token in preds:
                label_idxs = pred_token.nonzero(as_tuple=True)
                sent_labels.append([self.idx2label[int(i)] for i in label_idxs[0]])
            pred_labels.append(sent_labels)
        return pred_labels

    def test(self, dev, cutoff=0.5):
        text = [tweet["text"] for tweet in dev.values()]
        gold = [[tuple(label) for label in tweet["labels"]] for tweet in dev.values()]
        flat_gold = [l for t in gold for l in t]

        # binarize gold data
        mlb = MultiLabelBinarizer()
        mlb.fit(flat_gold)
        b_gold = mlb.transform(flat_gold)

        output = self.predict(text, cutoff)
        output = [[tuple(l) for l in sent] for sent in output]
        flat_output = [l for t in output for l in t]
        b_pred = mlb.transform(flat_output)

        acc = accuracy_score(b_gold, b_pred)
        f1 = f1_score(b_gold, b_pred, average="micro")
        return acc, f1






class MulticlassLSTM(nn.Module):
    def __init__(self, word2idx,
                 embedding_matrix,
                 label2idx,
                 embedding_dim,
                 hidden_dim,
                 num_layers=2,
                 lstm_dropout=0.2,
                 word_dropout=0.5,
                 train_embeddings=False):
        super(MulticlassLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2idx)
        self.tagset_size = len(label2idx)
        self.label2idx = label2idx
        self.idx2label = dict([(v, k) for k, v in label2idx.items()])
        self.lstm_dropout = lstm_dropout
        self.word_dropout = word_dropout
        self.sentiment_criterion = nn.BCEWithLogitsLoss()
        self.loss_function = nn.BCEWithLogitsLoss(reduction='none')

        weight = torch.FloatTensor(embedding_matrix)
        self.word_embeds = nn.Embedding.from_pretrained(weight, freeze=False)
        self.word_embeds.requires_grad = train_embeddings
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)
    def forward(self, x):
        out = self.word_embeds(x)
        out, _ = self.lstm(out)
        out = self.hidden2tag(out)
        return out
    def get_loss(self, sents, targets):
        out = self.forward(sents)
        loss = self.loss_function(out, targets)
        # mask out loss for padding
        loss[targets == 20] = 0
        return loss.mean()
    def pred(self, sents, cutoff=0.05):
        pred_labels = []
        out = self.forward(sents)
        out = torch.sigmoid(out)
        preds = out > cutoff

        for pred_sent, org_sent in zip(preds, sents):
            sent_labels = []
            for pred_token, org_token in zip(pred_sent, org_sent):
                print(org_token)
                if org_token != 1:
                    label_idxs = pred_token.nonzero(as_tuple=True)
                    sent_labels.append([model.idx2label[int(i)] for i in label_idxs[0]])
            pred_labels.append(sent_labels)
        return pred_labels


#sys.path.append("..")
#from labelling_functions_baseline import label_match


"""
x = torch.LongTensor([[1,2], [3,4]])
gold = torch.Tensor([[[1,0,0,0], [0,1,0,1]], [[1,1,0,0], [0,1,1,1]]])

emb = nn.Embedding(2000, 300)
lstm = nn.LSTM(300, 50)
linear = nn.Linear(50, 4)

optimizer = torch.optim.Adam(list(emb.parameters()) + \
                             list(lstm.parameters()) + \
                             list(linear.parameters())
                             )

loss_function = nn.BCEWithLogitsLoss()

for epoch in range(50):
    emb.zero_grad()
    lstm.zero_grad()
    linear.zero_grad()

    out = emb(x)
    out, _ = lstm(out)
    out = linear(out)

    l = loss_function(out, gold)
    print(l.data)
    l.backward()

    optimizer.step()
"""

class Split(object):
    def __init__(self, texts, labels, vocab):
        texts = [torch.LongTensor(vocab.ws2ids(t)) for t in texts]
        self.data = list(zip(texts, labels))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def pack_words(self, ws):
        return pack_sequence(ws)
    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda item : len(item[0]), reverse=True)
        words = pad_sequence([w for w,_ in batch],
                             padding_value=vocab[vocab.pad],
                             batch_first=True)
        #words = pack_sequence(words)
        targets = pad_sequence([t for _,t in batch],
                               padding_value=20,
                               batch_first=True)
        return words, targets


def get_iter(split_file, vocab):
    with open(split_file) as o:
        train_dict = json.load(o)
    train = list(train_dict.values())
    texts = [tweet["text"] for tweet in train]
    gold = [[tuple(label) for label in tweet["labels"]] for tweet in train]
    y = convert_gold(gold, label2idx)
    data_iter = Split(texts, y, vocab)
    return data_iter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_LAYERS", "-nl", default=1, type=int)
    parser.add_argument("--EPOCHS", "-e", default=10, type=int)
    parser.add_argument("--HIDDEN_DIM", "-hd", default=100, type=int)
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
    parser.add_argument("--EMBEDDING_DIM", "-ed", default=100, type=int)
    parser.add_argument("--TRAIN_EMBEDDINGS", "-te", action="store_false")
    parser.add_argument("--EMBEDDINGS", "-emb",
                        default="/home/jeremy/Exps/embeddings/norwegian/124/model.txt")

    args = parser.parse_args()
    print(args)

    label_match = {"pron": "pron",
                   "pronoun_subject": "pronoun_subject",
                   "pronount_subject": "pronoun_subject",
                   "high_recall_pronoun": "",
                   "high_prec_pronoun_subject": "pronoun_subject",
                   "pronoun_object": "pronoun_object",
                   "high_prec_pronoun_object": "pronoun_object",
                   "copula": "copula",
                   "copula_near_pronoun": "copula",
                   "no_other_verb_in_sent_copula": "copula",
                   "incompatible_forms_copula": "copula",
                   "det_e_copula": "copula",
                   "contraction": "contraction",
                   "any_apostrofe": "",
                   "any_but_tn": "",
                   "no_apostrofe_neg_contractions": "",
                   "no_apostrofe_rru_contractions": "",
                   "present_marker_deletion": "present_marker_deletion",
                   "present_marker_deletion_ordsboka": "",
                   "present_marker_near_aux": "",
                   "present_marker_near_pron": "",
                   "present_marker_near_aux_pron": "",
                   "apocope_verbs": "apocope",
                   "apocope_nouns": "apocope",
                   "adjective_declension": "adjectival_declension",
                   "adjectival_declension": "adjectival_declension",
                   "nominal_declension": "nominal_declension",
                   "conjugation": "conjugation",
                   "dem_pro": "demonstrative_pronoun",
                   "dem_pro_names": "demonstrative_pronoun",
                   "h_v": "h_v",
                   "gender": "gender",
                   "shortening": "shortening",
                   "functional": "functional",
                   "marked": "marked",
                   "phonemic_spelling": "phonemic_spelling",
                   "interjection": "interjection",
                   "voicing": "voicing",
                   "vowel_shift": "vowel_shift",
                   "palatalization": "palatalization",
                   "lexical": "lexical",
                   "apocope": "apocope",
                   "demonstrative_pronoun": "demonstrative_pronoun"
                   }

    """
    # Get embeddings (CHANGE TO GLOVE OR FASTTEXT EMBEDDINGS)
    embeddings = WordVecs(args.EMBEDDINGS)
    print("loaded embeddings from {0}".format(args.EMBEDDINGS))
    w2idx = embeddings._w2idx

    # Create shared vocabulary for tasks
    vocab = Vocab(train=True)

    # Update with word2idx from pretrained embeddings so we don't lose them
    # making sure to change them by one to avoid overwriting the UNK token
    # at index 0 and <pad> at 1
    with_unk = {}
    for word, idx in embeddings._w2idx.items():
        with_unk[word] = idx + 2
    vocab.update(with_unk)

    label2idx = {k: i for i, k in enumerate(label_match.keys())}

    train_iter = get_iter("../test_data/train.json", vocab)
    dev_iter = get_iter("../test_data/dev.json", vocab)

    train_loader = DataLoader(train_iter,
                              batch_size=args.BATCH_SIZE,
                              collate_fn=train_iter.collate_fn,
                              shuffle=True)
    dev_loader = DataLoader(dev_iter,
                            batch_size=args.BATCH_SIZE,
                            collate_fn=dev_iter.collate_fn,
                            shuffle=False)

    diff = len(vocab) - embeddings.vocab_length - 1
    UNK_embedding = np.zeros((1, args.EMBEDDING_DIM))
    new_embeddings = np.zeros((diff, args.EMBEDDING_DIM))
    new_matrix = np.concatenate((UNK_embedding, embeddings._matrix, new_embeddings))


    model = MulticlassLSTM(vocab,
                           new_matrix,
                           label2idx,
                           args.EMBEDDING_DIM,
                           args.HIDDEN_DIM,
                           num_layers=2,
                           lstm_dropout=0.2,
                           word_dropout=0.5,
                           train_embeddings=False)

    optim = torch.optim.Adam(model.parameters())

    for epoch in range(args.EPOCHS):
        print(f"epoch {epoch+1}")
        batch_losses = 0
        num_batches = 0

        for sents, targets in tqdm(train_loader):
            model.zero_grad()

            loss = model.get_loss(sents, targets)
            batch_losses += loss.data
            num_batches += 1

            loss.backward()
            optim.step()
        print()
        print("loss: {0:.3f}".format(batch_losses / num_batches))
    """

    label2idx = {k: i for i, k in enumerate(label_match.keys())}
    with open("../test_data/train.json") as o:
        train = json.load(o)
    with open("../test_data/dev.json") as o:
        dev = json.load(o)

    train_x = [t["text"] for t in train.values()]
    train_labels = [t["labels"] for t in train.values()]
    train_y = convert_gold(train_labels, label2idx)

    mb = MulticlassBert(label2idx)

    mb.train_model(train_x[:40], train_y[:40], dev, batch_size=5)
