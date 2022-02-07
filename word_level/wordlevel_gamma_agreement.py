from pygamma_agreement import Continuum, CombinedCategoricalDissimilarity
from pyannote.core import Segment
from time import time
import csv
import os

def create_continuum(raw_ann_dir1, raw_ann_dir2,
                     ann1="alexandra", ann2="marie"):
    continuum = Continuum()
    current_add_idx = 0
    for filename in os.listdir(raw_ann_dir1):
        max_idx = current_add_idx
        if ".ann" in filename:
            ann1_file = os.path.join(raw_ann_dir1, filename)
            ann2_file = os.path.join(raw_ann_dir2, filename)
            for line in open(ann1_file):
                if not line.startswith("#"):
                    ann = line.split("\t")[1]
                    label, bidx, eidx = ann.split()
                    bidx = int(bidx) + current_add_idx
                    eidx = int(eidx) + current_add_idx
                    if eidx > max_idx:
                        max_idx = eidx
                    continuum.add(ann1, Segment(bidx, eidx), label)
            for line in open(ann2_file):
                if not line.startswith("#"):
                    ann = line.split("\t")[1]
                    label, bidx, eidx = ann.split()
                    bidx = int(bidx) + current_add_idx
                    eidx = int(eidx) + current_add_idx
                    if eidx > max_idx:
                        max_idx = eidx
                    continuum.add(ann2, Segment(bidx, eidx), label)
        current_add_idx = max_idx + 10
    return continuum

continuum = create_continuum("IAA/alexandra",
                             "IAA/marie")

dissim = CombinedCategoricalDissimilarity()
gamma_results = continuum.compute_gamma(dissim)

print("Gamma: {0:.3f}".format(gamma_results.gamma))
