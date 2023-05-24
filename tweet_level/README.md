# NorDial

This repository contains the data and models described in following paper: [NorDial: A Preliminary Corpus of Written Norwegian Dialect Use](https://aclanthology.org/2021.nodalida-main.51/).

## Paper Abstract

Norway has a large amount of dialectal
variation, as well as a general tolerance
to its use in the public sphere. There are,
however, few available resources to study
this variation and its change over time and
in more informal areas, e.g. on social media. In this paper, we propose a first step
to creating a corpus of dialectal variation
of written Norwegian. We collect a small
corpus of tweets and manually annotate
them as Bokmål, Nynorsk, any dialect, or
a mix. We further perform preliminary experiments with state-of-the-art models, as
well as an analysis of the data to expand
this corpus in the future. Finally, we make
the annotations available for future work


## Dataset

The data is found in the 'data' repository, which contains a train, dev, and test json. The JSON format for each tweet contains keys for 'sent_id', 'text', and 'category', which can be 'bokmål', 'nynorsk', 'dialectal', or 'mixed'

```
    {
        "sent_id": "1334076729695100928",
        "text": "@kiheger Eg skal bidra med 700 kroner i ekstra etter heving av frikortgrensa",
        "category": "nynorsk"
    }
```

## Code to train NB-BERT dialect model

The code to train the best dialect prediction model from the paper is available in the experiments directory. After making sure to have the required libraries (see below), you can train the model in the following way:

```
cd experiments
python3 bert_classifier.py --model NbAiLab/nb-bert-base --train --test
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
