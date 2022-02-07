from collections import defaultdict

# A small function to check overlap between two labels for the word-level
# annotation scheme of the NorDial-project.
# For problems, contact Petter Mæhlum,  pettemae[alfakrøll]ifi.uio.no
# The first part is a file-reading version, the second is a free-standing version.

# Initiating the overlap dictionary
overlaps = defaultdict(dict)

# Opening file. Predefined filename is "overlap.csv" which should be located in the same folder.
with open("overlap.csv","r",encoding="utf-8") as csv_file:
    a = next(csv_file)
    all_labels = a.split(",")
    while a:
        a = next(csv_file,"")
        info = a.split(",")
        for num, label in enumerate(info[1:]):
            if label != "" and label != "\n":
                overlaps[info[0]][all_labels[(num+1)]] = label
            
def has_overlap(label1,label2):
    """
    Returns TRUE if the two labels overlap
    Otherwise returns FALSE
    """
    label1,label2 = label1.lower(),label2.lower()
    return overlaps[label1].get(label2,False)


# Testing the first function
def runtest1():
    assert has_overlap("pron-obj","pron-subj")
    assert has_overlap("present_marker_deletion","gender")
    assert has_overlap("present_marker_deletion","dem_pro")
    assert has_overlap("copula","pron-subj")
    assert has_overlap("lexical","marked")
    assert has_overlap("lexical","functional")
    assert not has_overlap("gender","nominal_declension")
    assert not has_overlap("present_marker_deletion","conjugation")

runtest1()

############################################################
# A compact but sufficient version not requiring reading   #
# but a bit clunky                                         #
############################################################

def has_overlap2(label1,label2):
    """
    Returns TRUE if the two labels overlap
    Otherwise returns FALSE
    """
    label1,label2 = label1.lower(),label2.lower()
    labeloverlaps = {"pron-subj":{'pron-obj': '1', 'copula': '2', 'present_marker_deletion': '4', 'lexical': '6', 'dem_pro': '9', 'gender': '14', 'marked': '19', 'adjectival_declension': '23', 'nominal_declension': '28', 'conjugation': '34', 'functional': '40', 'interjection\n': '51\n'},
    "pron-obj":{'pron-subj': '1', 'copula': '3', 'present_marker_deletion': '5', 'lexical': '7', 'dem_pro': '10', 'gender': '15', 'marked': '20', 'adjectival_declension': '24', 'nominal_declension': '29', 'conjugation': '35', 'functional': '41', 'interjection\n': '52\n'},
    "copula":{'pron-subj': '2', 'pron-obj': '3', 'lexical': '8', 'dem_pro': '11', 'gender': '16', 'marked': '21', 'adjectival_declension': '25', 'nominal_declension': '30', 'conjugation': '36', 'functional': '42', 'interjection\n': '53\n'},
    "present_marker_deletion":{'pron-subj': '4', 'pron-obj': '5', 'dem_pro': '12', 'gender': '17', 'adjectival_declension': '26', 'nominal_declension': '31', 'functional': '43', 'interjection\n': '54\n'},
    "lexical":{'pron-subj': '6', 'pron-obj': '7', 'copula': '8', 'dem_pro': '13', 'marked': '63', 'functional': '44', 'interjection\n': '55\n'},
    "dem_pro":{'pron-subj': '9', 'pron-obj': '10', 'copula': '11', 'present_marker_deletion': '12', 'lexical': '13', 'gender': '18', 'marked': '22', 'adjectival_declension': '27', 'nominal_declension': '32', 'conjugation': '37', 'functional': '45', 'interjection\n': '56\n'},
    "gender":{'pron-subj': '14', 'pron-obj': '15', 'copula': '16', 'present_marker_deletion': '17', 'dem_pro': '18', 'functional': '46', 'interjection\n': '57\n'},
    "marked":{'pron-subj': '19', 'pron-obj': '20', 'copula': '21', 'lexical': '63', 'dem_pro': '22', 'functional': '47', 'interjection\n': '58\n'},
    "adjectival_declension":{'pron-subj': '23', 'pron-obj': '24', 'copula': '25', 'present_marker_deletion': '26', 'dem_pro': '27', 'nominal_declension': '33', 'conjugation': '38', 'functional': '48', 'interjection\n': '59\n'},
    "nominal_declension":{'pron-subj': '28', 'pron-obj': '29', 'copula': '30', 'present_marker_deletion': '31', 'dem_pro': '32', 'adjectival_declension': '33', 'conjugation': '39', 'functional': '49', 'interjection\n': '60\n'},
    "conjugation":{'pron-subj': '34', 'pron-obj': '35', 'copula': '36', 'dem_pro': '37', 'adjectival_declension': '38', 'nominal_declension': '39', 'functional': '50', 'interjection\n': '61\n'},
    "functional":{'pron-subj': '40', 'pron-obj': '41', 'copula': '42', 'present_marker_deletion': '43', 'lexical': '44', 'dem_pro': '45', 'gender': '46', 'marked': '47', 'adjectival_declension': '48', 'nominal_declension': '49', 'conjugation': '50', 'interjection\n': '62\n'},
    "interjection":{'pron-subj': '51', 'pron-obj': '52', 'copula': '53', 'present_marker_deletion': '54', 'lexical': '55', 'dem_pro': '56', 'gender': '57', 'marked': '58', 'adjectival_declension': '59', 'nominal_declension': '60', 'conjugation': '61', 'functional': '62'}}
    return labeloverlaps[label1].get(label2,False)

# Testing the second function
def runtest2():
    assert has_overlap2("pron-obj","pron-subj")
    assert has_overlap2("present_marker_deletion","gender")
    assert has_overlap2("present_marker_deletion","dem_pro")
    assert has_overlap2("copula","pron-subj")
    assert has_overlap2("lexical","marked")
    assert has_overlap2("lexical","functional")
    assert not has_overlap2("gender","nominal_declension")
    assert not has_overlap2("present_marker_deletion","conjugation")



if __name__ == "__main__":
    runtest1()
    runtest2()
    print("All tests ran without issues.")
