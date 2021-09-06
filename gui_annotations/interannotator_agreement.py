import json
import krippendorff
import sys

def all_same(items):
    return all(x == items[0] for x in items)

label_dict = {"bokmål": 0, "nynorsk": 1, "dialectal": 2, "mixed": 3}

if __name__ == "__main__":
    all_data = []
    all_idxs = []
    for file in sys.argv[1:]:
        print(file)
        with open(file) as o:
            d = json.load(o)
            idxs = sorted([int(k) for k in d.keys()])
            original_labels = [d[str(idx)]["category"] for idx in idxs]
            new_labels = [d[str(idx)]["corrected_category"] for idx in idxs]
            final_labels = []
            for original, new in zip(original_labels, new_labels):
                if new in ["", "NONE"]:
                    final_labels.append(label_dict[original])
                else:
                    final_labels.append(label_dict[new])
            all_data.append(final_labels)
            all_idxs.append(idxs)

    assert all_same(all_idxs), "not all ids match up"

    alpha = krippendorff.alpha(all_data, level_of_measurement="nominal")
    print("Kripp Alpha: {0:.3f}".format(alpha))
