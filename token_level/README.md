# NorDial

This repository contains the data and models described in following paper: [Identifying Token-Level Dialectal Features in Social Media](https://aclanthology.org/2023.nodalida-1.16/).

## Paper Abstract

Dialectal variation is present in many human languages and is attracting a growing interest in NLP. Most previous work concentrated on either (1) classifying dialectal varieties at the document or sentence level or (2) performing standard NLP tasks on dialectal data. In this paper, we propose the novel task of token-level dialectal feature prediction. We present a set of fine-grained annotation guidelines for Norwegian dialects, expand a corpus of dialectal tweets, and manually annotate them using the introduced guidelines. Furthermore, to evaluate the learnability of our task, we conduct labeling experiments using a collection of baselines, weakly supervised and supervised sequence labeling models. The obtained results show that, despite the difficulty of the task and the scarcity of training data, many dialectal features can be predicted with reasonably high accuracy.


## Dataset

The data is found in the 'data' repository, which contains a train, dev, and test json. The JSON format contains a nested dictionary for each tweet, with a sentence id. Each tweet contains the original non-tokenized text (fulltext), the tokenized text needed for annotating token-level (text), and the labels, which contain a list of lists, as each token can have more than one possible label.

```
    {"1550_raw": {
                    "fulltext": "Greie nesten ikkje vente p\u00e5 at Tour de France skal starte # 2tdf", 
                    "text": ["Greie", "nesten", "ikkje", "vente", "p\u00e5", "at", "Tour", "de", "France", "skal", "starte", "#", "2tdf"], 
                    "labels": [["present_marker_deletion"], [], [], [], [], [], [], [], [], [], [], [], []]
                 },
    "1538_raw":  {
                    "fulltext": "Tour de Norge . 1 . Hushovd 2 . Boasson Hagen 3 . Hesjedal . To norske og ein hall norsk ! # fantastisk", 
                    "text": ["Tour", "de", "Norge", ".", "1", ".", "Hushovd", "2", ".", "Boasson", "Hagen", "3", ".", "Hesjedal", ".", "To", "norske", "og", "ein", "hall", "norsk", "!", "#", "fantastisk"], 
                    "labels": [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ["phonemic_spelling"], [], [], [], []]},
    }
```

## Code to train BERT token-level dialect identification model

The code to train the weak supervision model can be found in ```./experiments```.

```
cd experiments
python3 train_bertmodel.py --model NbAiLab/nb-bert-base --train --test
```

### Requirements

1. pytorch
2. transformers library
3. matplotlib
4. seaborn
5. pandas
6. sklearn
7. tqdm

# Terms of use
The data is distributed under a Creative Commons Attribution-NonCommercial licence (CC BY-NC 4.0), access the full license text here: https://creativecommons.org/licenses/by-nc/4.0/

The licence is motivated by the need to block the possibility of third parties redistributing the orignal reviews for commercial purposes. Note that **machine learned models**, extracted **lexicons**, **embeddings**, and similar resources that are created on the basis of the corpus are not considered to contain the original data and so **can be freely used also for commercial purposes** despite the non-commercial condition.
