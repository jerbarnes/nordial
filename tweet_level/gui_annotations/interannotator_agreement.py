import json
import krippendorff
import sys
import argparse

def all_same(items):
    return all(x == items[0] for x in items)

label_dict = {"bokmål": 0, "nynorsk": 1, "dialectal": 2, "mixed": 3}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_files", nargs="+")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    all_data = []
    all_idxs = []
    disagreements = []
    for file in args.annotation_files:
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
    print()
    print("Kripp Alpha: {0:.3f}".format(alpha))

    if args.verbose:
        already_printed = []
        for d1, idx1, file1 in zip(all_data, all_idxs, sys.argv[1:]):
            for d2, file2 in zip(all_data, sys.argv[1:]):
                tup = sorted((d1, d2))
                if d1 is not d2 and tup not in already_printed:
                    already_printed.append(tup)
                    print()
                    print("Disagreements between {} and {}".format(file1, file2))
                    print("=" * 80)
                    print()
                    disagree_idxs = []
                    for i in range(len(d1)):
                        if d1[i] != d2[i]:
                            disagree_idxs.append(idx1[i])
                    with open(file1) as o:
                        lfile1 = json.load(o)
                    with open(file2) as o:
                        lfile2 = json.load(o)
                    for idx in disagree_idxs:
                        print("## text: {}".format(lfile1[str(idx)]["text"]))
                        print("## original: {}".format(lfile1[str(idx)]["category"]))
                        print("## corrected 1: {}".format(lfile1[str(idx)]["corrected_category"]))
                        print("## corrected 2: {}".format(lfile2[str(idx)]["corrected_category"]))
                        #print("-" * 40)
                        print()
                    print("=" * 80)
