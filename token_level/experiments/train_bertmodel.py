import json
import torch
import os
from datasets import Dataset, load_metric
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

import argparse

def read_file(path):
    with open(path) as input_data:
        content = json.load(input_data)
    return(content)

def get_data(split, train_data="train.json"):

    path_files = "../test_data/"

    if split == "train":
        datadict = read_file(path_files+train_data)
    elif split == "dev":
        datadict = read_file(path_files+"dev.json")
    elif split == "test":
        datadict = read_file(path_files+"test.json")

    examples = []

    # Use merged labels as a proxy for multiclass token classification
    with open('merged_classes.json') as merged_c_files:
        merged = json.load(merged_c_files)

    # build the label to idx dictionary
    # first label is out
    all_labels = ['', 'adjectival_declension', 'apocope', 'conjugation', 'contraction', 'copula', 'demonstrative_pronoun', 'functional', 'gender', 'h_v', 'interjection', 'lexical', 'marked', 'nominal_declension', 'palatalization', 'phonemic_spelling', 'present_marker_deletion', 'pronoun_object', 'pronoun_subject', 'shortening', 'voicing', 'vowel_shift', 'adj_dec_apocope', 'adj_dec_contr', 'adj_dec_contr_voic', 'adj_dec_lex', 'adj_decl_palat', 'adj_decl_phonem', 'adj_decl_voic', 'adj_decl_vowel_shift', 'apoco_conju', 'apoco_contra', 'apoco_copu', 'apoco_copu_short', 'apoco_copu_vowel_shift', 'apoco_func', 'apoco_func_pala', 'apoco_func_vowel_shift', 'apoco_nom_dec', 'apoco_pala', 'apoco_pala_vowel_shift', 'apoco_phon_spell', 'apoco_vowel_shift', 'conju_contra', 'conju_contra_funct_vowel_shift', 'conju_copu', 'conju_copu_vowel_shift', 'conju_funct', 'conju_lexi', 'conju_marked', 'conju_pala', 'conju_pala_vowel_shift', 'conju_phon', 'conju_pml', 'conju_pml_voicing', 'conju_pml_vowel_shift', 'conju_short', 'conju_short_vowel_shift', 'conju_voicing', 'conju_vowel_shift', 'contra', 'contra_copu', 'contra_copu_func', 'contra_copu_phon_spelling', 'contra_demon_pronoun', 'contra_demon_pronoun_func', 'contra_func', 'contra_func_marked', 'contra_func_pml', 'contra_func_pml_vowel_shift', 'contra_func_pron_subj', 'contra_func_pron_subj_vowel_shift', 'contra_h_v', 'contra_marked', 'contra_nom_dec', 'contra_pala', 'contra_pala_phon', 'contra_pala_pml_vowel_shift', 'contra_phon', 'contra_phon_pron_subj', 'contra_pml', 'contra_pron_obj', 'contra_pron_subj', 'contra_pron_subj_shortening', 'contra_shortening', 'contra_voic', 'contra_vowel_shift', 'copula_phon', 'copula_pml', 'copula_short_vowel_shift', 'copula_shortening', 'copula_vowel_shift', 'dem_pron_shortening', 'func_h_v', 'func_h_v_vowel_shift', 'func_lexi', 'func_lexi_shortening', 'func_marked', 'func_pala', 'func_phon', 'func_phon_vowel_shift', 'func_pml', 'func_pron_obj', 'func_pron_subj', 'func_pron_subj_vowel_shift', 'func_shortening', 'func_voic', 'func_voic_vowel_shift', 'func_vowel_shift', 'h_v_nom_dec', 'h_v_pala_shortening', 'h_v_pron_subj', 'h_v_shortening', 'h_v_vowel_shift', 'interj_phon', 'interj_vowel_shift', 'lexi_marked', 'lexi_nom_dec', 'lexi_phon', 'lexi_pml', 'lexi_voicing', 'marked_phon', 'marked_pml', 'marked_pml_vowel_shift', 'marked_shortening', 'marked_voic', 'marked_vowel_shift', 'nom_dec_pala', 'nom_dec_pala_vowel_shift', 'nom_dec_phon', 'nom_dec_phon_vowel_shift', 'nom_dec_short', 'nom_dec_short_vowel_shift', 'nom_dec_voic_vowel_shift', 'nom_dec_voicing', 'nom_dec_vowel_shift', 'pala_phon', 'pala_pml_vowel_shift', 'pala_pron_obj', 'pala_pron_subj', 'pala_short', 'pala_short_vowel_shift', 'pala_voic', 'pala_vowel_shift', 'phon_pml', 'phon_pron_obj', 'phon_pron_subj', 'phon_shortening', 'phon_voic', 'phon_vowel_shift', 'pml_shortening', 'pml_voic', 'pml_voic_vowel_shift', 'pml_vowel_shift', 'pron_obj_vowel_shift', 'pron_subj_vowel_shift', 'shortening_voic', 'shortening_vowel_shift', 'voic_vowel_shift']

    merged_labels_to_idxs = dict([(label, idx) for idx, label in enumerate(all_labels)])

    for idx in datadict:
        tokens = [t for t in datadict[idx]["text"]]

        for nm in merged:
            new_label = merged[nm]["merged"]
            for label in datadict[idx]["labels"]:
                if len(label) > 1:
                    old_label = label
                    if sorted(old_label) == merged[nm]['original']:
                        datadict[idx]["labels"][datadict[idx]["labels"].index(label)] = [new_label]
        l_list = [''.join(l) for l in datadict[idx]["labels"]]
        l_list = [merged_labels_to_idxs[l] for l in l_list]
        examples.append({"id": idx, "tokens": tokens, "labels": l_list})

    return examples, merged_labels_to_idxs


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    flat_preds = [i for l in true_predictions for i in l]
    flat_labels = [i for l in true_labels for i in l]

    prec = precision_score(flat_labels, flat_preds, average="micro", labels=label_list[1:])
    rec = recall_score(flat_labels, flat_preds, average="micro", labels=label_list[1:])
    f1 = f1_score(flat_labels, flat_preds, average="micro", labels=label_list[1:])
    acc = accuracy_score(flat_labels, flat_preds)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
    }

def eval_model(dataset):
    predictions, labels, _ = trainer.predict(dataset)
    predictions = np.argmax(predictions, axis=2)
    #
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    #
    flat_preds = [i for l in true_predictions for i in l]
    flat_labels = [i for l in true_labels for i in l]
    #
    f1 = f1_score(flat_labels, flat_preds, average="micro", labels=label_list[1:])
    return f1, true_labels, true_predictions

def preds_to_json(test, gold, preds, output_file):
    final = []
    for ex, g, p in zip(test, gold, preds):
        ex["labels"] = g
        ex["pred"] = p
        final.append(ex)
    with open(output_file, "w") as out:
        json.dump(final, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="ltgoslo/norbert")
    parser.add_argument("--train_data", default="train.json")
    parser.add_argument("--output_dir", default="norbert")

    args = parser.parse_args()
    print(args)


    # import the raw data and convert to examples and label2idx
    train_examples, train_l2idx = get_data("train", train_data=args.train_data)
    dev_examples, dev_l2idx = get_data("dev")
    test_examples, test_l2idx = get_data("test")

    num_labels = len(train_l2idx)
    label_list = list(train_l2idx.keys())


    # import bert model
    #model_name = "ltgoslo/norbert"
    #model_name = "NbAiLab/nb-bert-base"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # Convert to dataset format and tokenize using BERT tokenizer
    train = Dataset.from_list(train_examples, split="train")
    train_tokenized = train.map(tokenize_and_align_labels, batched=True)

    dev = Dataset.from_list(dev_examples, split="dev")
    dev_tokenized = dev.map(tokenize_and_align_labels, batched=True)

    test = Dataset.from_list(test_examples, split="test")
    test_tokenized = test.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    #load_best_model_at_end=True,
    save_total_limit=0,
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()




    dev_f1, dev_gold, dev_preds = eval_model(dev_tokenized)
    print()
    print("#"*20)
    print("DEV F1: {0:.1f}".format(dev_f1 * 100))
    output_file = os.path.join(args.output_dir, "dev_preds.json")
    preds_to_json(dev, dev_gold, dev_preds, output_file)


    f1, gold, preds = eval_model(test_tokenized)
    print()
    print("#"*20)
    print("Test F1: {0:.1f}".format(f1 * 100))
    output_file = os.path.join(args.output_dir, "preds.json")
    preds_to_json(test, gold, preds, output_file)