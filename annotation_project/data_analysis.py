import json
from collections import Counter


if __name__ == "__main__":


    # get basic statistics
    with open("dialect_train.json") as infile:
        train = json.load(infile)
    with open("dialect_dev.json") as infile:
        dev = json.load(infile)
    with open("dialect_test.json") as infile:
        test = json.load(infile)

    # Number of tweets in train, dev, test
    print("Num tweets:   Train - {0}  Dev - {1}  Test - {2}".format(len(train), len(dev), len(test)))

    # distribution of classes in each
    print("Train distribution: ")
    c = Counter()
    for d in train:
        c[d["category"]] += 1
    print(c)
    print()

    print("Dev distribution: ")
    c = Counter()
    for d in dev:
        c[d["category"]] += 1
    print(c)
    print()

    print("Test distribution: ")
    c = Counter()
    for d in test:
        c[d["category"]] += 1
    print(c)
    print()


    # avg. number of tokens (per category)


    #
