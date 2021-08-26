from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import json
import os
import torch
from transformers import AdamW
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import argparse

font = {'serif': ['Times'],
        'size': 16}

matplotlib.rc('font', **font)

def collate_fn(batch):
    batch = sorted(batch, key=lambda item : item[0]["attention_mask"].numpy().sum(), reverse=True)
    max_length = max([len(item[0]["input_ids"]) for item in batch])
    #print(max_length)
    new_input_ids = []
    new_tt = []
    new_att = []
    new_labels = []
    for info, label in batch:
        self_length = len(info["input_ids"])
        #print(self_length)
        padding = torch.zeros(max_length - self_length)
        new_input_ids.append(torch.cat((info["input_ids"], padding)).long())
        new_tt.append(torch.cat((info["token_type_ids"], padding)).long())
        new_att.append(torch.cat((info["attention_mask"], padding)).long())
        new_labels.append(label)
    new_batch = {"input_ids": torch.stack(new_input_ids),
                 "token_type_ids": torch.stack(new_tt),
                 "attention_mask": torch.stack(new_att)
                 }
    new_labels = torch.tensor(new_labels).long()
    return new_batch, new_labels

# create dataloader
def load_dataset(dataset_file, tokenizer):
    label_map = {"bokmål": 0, "nynorsk": 1, "dialectal": 2, "mixed": 3}
    final_data = []
    with open(dataset_file) as o:
        data = json.load(o)
    texts = [t["text"] for t in data]
    labels = [label_map[t["category"]] for t in data]
    tokenized = tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=True)
    for i in range(len(data)):
        info = {"input_ids": tokenized["input_ids"][i],
                "token_type_ids": tokenized["token_type_ids"][i],
                "attention_mask": tokenized["attention_mask"][i],
        }
        final_data.append((info, labels[i]))
    return final_data

def test_model(dataloader, model):
    model.eval()
    preds = []
    gold = []
    for info, label in tqdm(dataloader):
        output = model(**info)
        preds.extend(output.logits.detach().numpy().argmax(1))
        gold.extend(label.tolist())
    return gold, preds, f1_score(gold, preds, average="macro")


def train_model(trainloader, devloader, model, output_dir="saved_models", model_name="norbert", num_epochs=20):
    os.makedirs(output_dir, exist_ok=True)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    losses = []
    best_dev = 0
    for e in range(num_epochs):
        print("Epochs: {}".format(e+1))
        model.train()
        epoch_loss = 0
        for info, label in tqdm(trainloader):
            outputs = model(**info, labels=label)
            loss = outputs.loss
            loss.backward()
            epoch_loss += float(loss.detach())
            optimizer.step()
            #scheduler.step()
            model.zero_grad()
        epoch_loss /= len(trainloader)
        print("Loss: {0:.3f}".format(epoch_loss))
        losses.append(epoch_loss)

        # eval on dev
        dg, dp, dev_f1 = test_model(devloader, model)
        print("Dev F1: {0:.3f}".format(dev_f1))
        if dev_f1 > best_dev:
            print("New best model")
            model.save_pretrained(os.path.join("saved_models", model_name))
            best_dev = dev_f1

def get_errors(gold, pred, texts, idx2label):
    for g, p, t in zip(gold, pred, texts):
        if g != p:
            print(t)
            print("Gold: {0} - Pred: {1}".format(idx2label[g], idx2label[p]))
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NbAiLab/nb-bert-base")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    name_map = {"ltgoslo/norbert": "norbert",
                "NbAiLab/nb-bert-base": "nb-bert",
                "bert-base-multilingual-cased": "mbert"
                }
    short_name = name_map[args.model]

    print("importing data...")
    tokenizer = BertTokenizer.from_pretrained(args.model)
    train_data = load_dataset("../data/train.json",
                              tokenizer)
    train_loader = DataLoader(train_data,
                              batch_size=32,
                              shuffle=True,
                              collate_fn=collate_fn)

    dev_data = load_dataset("../data/dev.json",
                            tokenizer)
    dev_loader = DataLoader(dev_data,
                            batch_size=8,
                            shuffle=False,
                            collate_fn=collate_fn)

    test_data = load_dataset("../data/test.json",
                             tokenizer)
    test_loader = DataLoader(test_data,
                             batch_size=8,
                             shuffle=False,
                             collate_fn=collate_fn)

    print("importing {} model from {}...".format(short_name, args.model))
    if args.train:
        model = BertForSequenceClassification.from_pretrained(args.model, num_labels=4)

    if args.train:
        print("training model...")
        train_model(train_loader, dev_loader, model, model_name=short_name)

    print("evaluating model...")
    model = BertForSequenceClassification.from_pretrained(os.path.join("saved_models", short_name))
    idx2label = {0: 'bokmål', 1: 'nynorsk', 2: 'dialectal', 3: 'mixed'}
    dev_texts = [" ".join(tokenizer.convert_ids_to_tokens(i[0]['input_ids'][i[0]["input_ids"].nonzero()].squeeze())) for i in dev_data]
    dev_labels = [idx2label[i[1]] for i in dev_data]
    test_texts = [" ".join(tokenizer.convert_ids_to_tokens(i[0]['input_ids'][i[0]["input_ids"].nonzero()].squeeze())) for i in test_data]
    test_labels = [idx2label[i[1]] for i in test_data]

    dev_gold, dev_pred, dev_f1 = test_model(dev_loader, model)
    dev_prec = precision_score(dev_gold, dev_pred, average="macro")
    dev_rec = recall_score(dev_gold, dev_pred, average="macro")
    print("Dev Prec: {0:.3f}".format(dev_prec))
    print("Dev Rec: {0:.3f}".format(dev_rec))
    print("Dev F1: {0:.3f}".format(dev_f1))
    #dev_pred = [idx2label[i] for i in dev_pred]
    print()

    print("Dev confusion matrix")
    cm = confusion_matrix(dev_gold, dev_pred)
    print(cm)
    print()

    print("-" * 40)

    test_gold, test_pred, test_f1 = test_model(test_loader, model)
    test_prec = precision_score(test_gold, test_pred, average="macro")
    test_rec = recall_score(test_gold, test_pred, average="macro")
    print("Test Prec: {0:.3f}".format(test_prec))
    print("Test Rec: {0:.3f}".format(test_rec))
    print("Test F1: {0:.3f}".format(test_f1))
    #test_pred = [idx2label[i] for i in test_pred]
    print()

    print("Test confusion matrix")
    cm = confusion_matrix(test_gold, test_pred)
    print(cm)
    print()

    df = pd.DataFrame(cm, index=["BK", "NN", "DI", "MIX"], columns=["BK", "NN", "DI", "MIX"])
    cmap = sn.color_palette("crest", as_cmap=True)
    fig = sn.heatmap(df, annot=True, cmap=cmap, cbar=False)
    plt.show()

    get_errors(test_gold, test_pred, test_texts, idx2label)
