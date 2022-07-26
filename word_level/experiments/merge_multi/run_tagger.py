from nltk.corpus import brown
import numpy as np
import random
import os
import math
import json
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchtext.data import Example
from torchtext.data import Field, Dataset
from torchtext.data import BucketIterator
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score, f1_score
from bilstm_model import BiLSTMTagger
from conllu import parse

text_field = Field(sequential=True, tokenize=lambda x:x, include_lengths=True) # Default behaviour is to tokenize by splitting
label_field = Field(sequential=True, tokenize=lambda x:x, is_target=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def read_file(path):
    with open(path) as input_data:
        content = json.load(input_data)
    return(content)

def get_data(split):
    path_files = "../test_data/"

    if split == "train":
        dict = read_file(path_files+"train.json")
    elif split == "dev":
        dict = read_file(path_files+"dev.json")
    elif split == "test":
        dict = read_file(path_files+"test.json")

    examples = []
    fields = {"sentence_labels": ("labels", label_field),
              "sentence_tokens": ("text", text_field)}

    # examples = get_content(dict, examples, fields)
    for id in dict:
        tokens = [t for t in dict[id]["text"]]
        with open('merged_classes.json') as merged_c_files:
            merged = json.load(merged_c_files)
        for nm in merged:
            new_label = merged[nm]["merged"]
            for label in dict[id]["labels"]:
                if len(label) > 1:
                    old_label = label
                    # print(sorted(old_label))
                    if sorted(old_label) == merged[nm]['original']:
                        # print('yes')
                        # print(dict[id]["labels"][dict[id]["labels"].index(label)])
                        dict[id]["labels"][dict[id]["labels"].index(label)] = [new_label]
                        # print(dict[id]["labels"][dict[id]["labels"].index([new_label])])
        labels = [''.join(l) for l in dict[id]["labels"]]

        e = Example.fromdict({"sentence_labels": labels, "sentence_tokens": tokens}, fields=fields)
        examples.append(e)


    return Dataset(examples, fields=[('labels', label_field), ('text', text_field)])

def load_embeddings(path):
    """ Load the FastText embeddings from the embedding file. """
    print("Loading pre-trained embeddings")

    embeddings = {}
    maximum = 100000
    j = 0
    with open(path) as i:
        for line in i:
            j += 1
            if j <= maximum:
                if len(line) > 2:
                    line = line.strip().split()
                    word = line[0]
                    embedding = np.array(line[1:])
                    embeddings[word] = embedding
            else:
                break
    return embeddings

def initialize_embeddings(embeddings, vocabulary):
    """ Use the pre-trained embeddings to initialize an embedding matrix. """
    print("Initializing embedding matrix")
    embedding_size = len(embeddings["."])
    embedding_matrix = np.zeros((len(vocabulary), embedding_size), dtype=np.float32)

    for idx, word in enumerate(vocabulary.itos):
        if word in embeddings:
            embedding_matrix[idx,:] = embeddings[word]

    return embedding_matrix

def remove_predictions_for_masked_items(predicted_labels, correct_labels):

    predicted_labels_without_mask = []
    correct_labels_without_mask = []

    for p, c in zip(predicted_labels, correct_labels):
        if c > 1:
            predicted_labels_without_mask.append(p)
            correct_labels_without_mask.append(c)

    return predicted_labels_without_mask, correct_labels_without_mask

def train(model, train_iter, dev_iter, batch_size, max_epochs, num_batches, patience, output_path):
    print("Here!")
    criterion = nn.CrossEntropyLoss(ignore_index=1)  # we mask the <pad> labels
    optimizer = optim.Adam(model.parameters())

    train_f_score_history = []
    dev_f_score_history = []
    no_improvement = 0
    #print(train_iter)


    for epoch in range(max_epochs):
        #print(type(train_iter))
        #for batch in train_iter:
            #print(batch)

        total_loss = 0
        predictions, correct = [], []
        for batch in tqdm(train_iter, total=num_batches, desc=f"Epoch {epoch}"):

            optimizer.zero_grad()

            text_length, cur_batch_size = batch.text[0].shape

            pred = model(batch.text[0].to(device), batch.text[1].to(device)).view(cur_batch_size*text_length, NUM_CLASSES)
            gold = batch.labels.to(device).view(cur_batch_size*text_length)

            loss = criterion(pred, gold)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            _, pred_indices = torch.max(pred, 1)

            predicted_labels = list(pred_indices.cpu().numpy())
            correct_labels = list(batch.labels.view(cur_batch_size*text_length).numpy())

            predicted_labels, correct_labels = remove_predictions_for_masked_items(predicted_labels, correct_labels)

            predictions += predicted_labels
            correct += correct_labels

        train_scores = precision_recall_fscore_support(correct, predictions, average="micro")
        train_f_score_history.append(train_scores[2])

        print("Total training loss:", total_loss)
        print("Training performance:", train_scores)

        total_loss = 0
        predictions, correct = [], []
        for batch in dev_iter:

            text_length, cur_batch_size = batch.text[0].shape

            pred = model(batch.text[0].to(device), batch.text[1].to(device)).view(cur_batch_size * text_length, NUM_CLASSES)
            gold = batch.labels.to(device).view(cur_batch_size * text_length)
            loss = criterion(pred, gold)
            total_loss += loss.item()

            _, pred_indices = torch.max(pred, 1)
            predicted_labels = list(pred_indices.cpu().numpy())
            correct_labels = list(batch.labels.view(cur_batch_size*text_length).numpy())

            predicted_labels, correct_labels = remove_predictions_for_masked_items(predicted_labels, correct_labels)

            predictions += predicted_labels
            correct += correct_labels

        dev_scores = precision_recall_fscore_support(correct, predictions, average="micro")

        print("Total development loss:", total_loss)
        print("Development performance:", dev_scores)

        dev_f = dev_scores[2]
        if len(dev_f_score_history) > patience and dev_f < max(dev_f_score_history):
            no_improvement += 1

        elif len(dev_f_score_history) == 0 or dev_f > max(dev_f_score_history):
            print("Saving model.")
            torch.save(model, output_path)
            no_improvement = 0

        if no_improvement > patience:
            print("Development F-score does not improve anymore. Stop training.")
            dev_f_score_history.append(dev_f)
            break

        dev_f_score_history.append(dev_f)

    return train_f_score_history, dev_f_score_history

def test(model, test_iter, batch_size, labels, target_names):

    total_loss = 0
    predictions, correct = [], []
    for batch in test_iter:

        text_length, cur_batch_size = batch.text[0].shape

        pred = model(batch.text[0].to(device), batch.text[1].to(device)).view(cur_batch_size * text_length, NUM_CLASSES)
        gold = batch.labels.to(device).view(cur_batch_size * text_length)

        _, pred_indices = torch.max(pred, 1)
        predicted_labels = list(pred_indices.cpu().numpy())
        correct_labels = list(batch.labels.view(cur_batch_size*text_length).numpy())

        predicted_labels, correct_labels = remove_predictions_for_masked_items(predicted_labels, correct_labels)

        predictions += predicted_labels
        correct += correct_labels

    print(classification_report(correct, predictions, labels=labels, target_names=target_names))

    print("precision_recall_fscore_support = {}".format(precision_recall_fscore_support(correct, predictions, average='micro')))

    print("Micro F1 = {}".format(f1_score(correct, predictions, average='micro')))

    print('Accuracy = {}'.format(accuracy_score(correct, predictions)))

def main():

    train_data = get_data("train")
    dev_data = get_data("dev")
    test_data = get_data("test")

    print(train_data.fields)
    print(train_data[0].text)
    print(train_data[0].labels)

    print("Train:", len(train_data))
    print("Dev:", len(dev_data))
    print("Test:", len(test_data))

    # VOCAB_SIZE = 20000

    text_field.build_vocab(train_data)#, max_size=VOCAB_SIZE)
    label_field.build_vocab(train_data)#build_vocab(train_data)


    print(len(text_field.vocab))
    # print(len(text_field))
    print(len(label_field.vocab.itos))
    # print(text_field.vocab.stoi)
    print(label_field.vocab.stoi)
    #
    VOCAB_SIZE = int(len(text_field.vocab))
    BATCH_SIZE = 32

    train_iter = BucketIterator(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, sort_key=lambda x: len(x.text), sort_within_batch=True)

    dev_iter = BucketIterator(dataset=dev_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text), sort_within_batch=True)

    test_iter = BucketIterator(dataset=test_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text), sort_within_batch=True)

    EMBEDDING_PATH = ("/home/samia/Documents/INFO371_2022/bilestm_tagger/model.txt")

    embeddings = load_embeddings(EMBEDDING_PATH)
    embedding_matrix = initialize_embeddings(embeddings, text_field.vocab)
    embedding_matrix = torch.from_numpy(embedding_matrix).to(device)

    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    global NUM_CLASSES
    NUM_CLASSES = len(label_field.vocab)
    MAX_EPOCHS = 50
    PATIENCE = 3
    OUTPUT_PATH = "/tmp/bilstmtagger"
    num_batches = math.ceil(len(train_data) / BATCH_SIZE)

    tagger = BiLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE+2, NUM_CLASSES, embeddings=embedding_matrix)

    train_f, dev_f = train(tagger.to(device), train_iter, dev_iter, BATCH_SIZE, MAX_EPOCHS, num_batches, PATIENCE, OUTPUT_PATH)

    tagger = torch.load(OUTPUT_PATH)
    tagger.eval()
    labels = label_field.vocab.itos[3:]
    labels = sorted(labels, key=lambda x: x.split("-")[-1])
    label_idxs = [label_field.vocab.stoi[l] for l in labels]
    print('-'*60)
    test(tagger, test_iter, BATCH_SIZE, labels = label_idxs, target_names = labels)

if __name__ == '__main__':
    main()
