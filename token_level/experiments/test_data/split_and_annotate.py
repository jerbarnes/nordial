import json
import nltk


def collect_stats(data):
    fd = nltk.FreqDist()
    for key, value in data.items():
        fd.update([label for labels in value["labels"] for label in labels])

    for label, count in fd.most_common():
        rel_count = count / sum(fd.values()) * 100
        print(f"{label:>25} {count:>5} {rel_count:.1f}")
    return fd


with open("output_annotation_allesammen.json") as o:
    data = json.load(o)

# collect statistics about the label distribution
overall = collect_stats(data)

# create train/dev/test split where this distribution is maintained
test_keys = list(data.keys())[:500]
dev_keys = list(data.keys())[500:800]
train_keys = list(data.keys())[800:]

test = {k: v for k, v in data.items() if k in test_keys}
dev = {k: v for k, v in data.items() if k in dev_keys}
train = {k: v for k, v in data.items() if k in train_keys}

print("TEST")
test_df = collect_stats(test)
print("-" * 40)
print()

print("DEV")
dev_df = collect_stats(dev)
print("-" * 40)
print()

print("TRAIN")
train_df = collect_stats(train)

with open("train.json", "w") as o:
    json.dump(train, o)

with open("dev.json", "w") as o:
    json.dump(dev, o)

with open("test.json", "w") as o:
    json.dump(train, o)

# annotate the test split for geolect
# keep notes on the ones I'm not sure of

for key, value in test.items():
    print(value["fulltext"])
    value["geolect"] = []
    geo = input("dialect: ")
    geo = geo.split()
    if "o" in geo:
        value["geolect"].append("øst")
    elif "v" in geo:
        value["geolect"].append("vest")
    elif "t" in geo:
        value["geolect"].append("trøndersk")
    elif "n" in geo:
        value["geolect"].append("nord")

