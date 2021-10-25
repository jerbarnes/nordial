from skweak.base import SpanAnnotator, CombinedAnnotator
from skweak.heuristics import FunctionAnnotator
from skweak.aggregation import HMM
import os
from spacy.tokens import Doc #type: ignore
from typing import Sequence, Tuple, Optional, Iterable
from collections import defaultdict




class NordialAnnotator(CombinedAnnotator):
    """Annotator of entities in documents, combining several sub-annotators (such as gazetteers,
    spacy models etc.). To add all annotators currently implemented, call add_all(). """

    def __init__(self, nlp, bokmal, nynorsk, functional_dict, marked_dict):
        super(NordialAnnotator, self).__init__()
        self.nlp = nlp
        self.bokmal = bokmal
        self.nynorsk = nynorsk
        self.functional_dict = functional_dict
        self.marked_dict = marked_dict

    def add_all(self):
        self.add_annotator(FunctionAnnotator("pron", dialect_pronoun))
        self.add_annotator(FunctionAnnotator("pron_subj", pron_subj))
        self.add_annotator(FunctionAnnotator("pron_obj", pron_obj))
        self.add_annotator(FunctionAnnotator("copula", copula))
        self.add_annotator(FunctionAnnotator("present_marker_deletion",
                                             present_marker_deletion))
        self.add_annotator(FunctionAnnotator("h_v", h_v))
        self.add_annotator(FunctionAnnotator("contraction", contraction))
        self.add_annotator(FunctionAnnotator("adjective_declension", adj_dec))
        self.add_annotator(FunctionAnnotator("nominal_declension", nom_dec))
        self.add_annotator(FunctionAnnotator("conjugation", conjugation))
        self.add_annotator(FunctionAnnotator("dem_pro", dem_pro))
        self.add_annotator(FunctionAnnotator("gender", gender))
        self.add_annotator(FunctionAnnotator("shortening", shortening))

        self.add_annotator(VoicingAnnotator("voicing",
                                            self.bokmal,
                                            self.nynorsk))
        self.add_annotator(FunctionalAnnotator("functional",
                                               self.functional_dict))
        self.add_annotator(MarkedAnnotator("marked",
                                           self.marked_dict))
        self.add_annotator(ApocopeAnnotator("apocope",
                                            self.nlp,
                                            self.bokmal,
                                            self.nynorsk))

####################################################################
# all dialectal forms of probouns
####################################################################
def dialect_pronoun(doc):
    forms = ["æ", "æg", "jæ", "jæi", "je", "ej", "mæ", "dæ", "hu", "ho", "honn", "hænne", "dåkk", "døkk", "døkker", "økk", "dom", "dæi", "døm", "dømm", "dæm", "demm", "di", "æm", "æmm"]
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.text.lower() in forms:
            yield i, i+1, "pron"
        i += 1



####################################################################
# pron-subj
####################################################################
def pron_subj(doc):
    obj_pron = ["æ", "æg", "jæ", "jæi", "je", "ej" "i", "mæ", "meg", "dæ", "ham", "hu", "ho", "honn", "henne", "hænne", "oss", "dåkk", "døkk", "døkker", "økk", "dem", "dom", "dæi", "døm", "dømm", "dæm", "demm", "di", "æm", "æmm"]
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.text.lower() in obj_pron and tok.dep_ == "nsubj":
            yield i, i+1, "pron-subj"
        i += 1

####################################################################
# pron-obj
####################################################################
def pron_obj(doc):
    subj_pron = ["jeg", "eg", "æ", "æg", "jæ", "jæi", "ej", "je", "i", "mæ", "dæ", "hu", "ho", "honn", "hænne", "me", "dåkk", "døkk", "døkker", "økk", "dom", "dæi", "døm", "dømm", "dæm", "demm", "di", "æm", "æmm"]
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.text.lower() in subj_pron and tok.dep_ == "obj":
            yield i, i+1, "pron-obj"
        i += 1

####################################################################
# copula
####################################################################
def copula(doc):
    forms = ["e", "værra"]
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.text.lower() in forms and tok.dep_ in ["xcomp", "cop"]:
                yield i, i+1, "copula"
        i += 1


####################################################################
# contraction
####################################################################
def contraction(doc):
    i = 0
    contractions = ["'kke", "ekje", "kje"]
    while i < len(doc):
        tok = doc[i]
        exceptions = ["kanskje", "skje"]
        for contraction in contractions:
            if tok.text.lower().endswith(contraction) and tok.text.lower() not in exceptions:
                yield i, i+1, "contraction"
        i += 1


####################################################################
# palitalization
####################################################################


####################################################################
# present_marker_deletion
####################################################################

def present_marker_deletion(doc):
    forms = ["ska", "vi"]
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.text in forms and tok.pos_ in ["AUX", "VERB"]:
                yield i, i+1, "present_marker_deletion"
        i += 1

def present_marker_deletion2(doc):
    """
    TODO: finish this for other verbs (velge -> vel)
    """

    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.text in forms and tok.pos_ in ["AUX", "VERB"]:
                yield i, i+1, "present_marker_deletion"
        i += 1

####################################################################
# apocope
####################################################################

class ApocopeAnnotator(SpanAnnotator):
    def __init__(self, name, nlp, bokmal, nynorsk):
        super(ApocopeAnnotator, self).__init__(name)
        self.nlp = nlp
        self.bokmal = bokmal
        self.nynorsk = nynorsk
    #
    def find_spans(self, doc):
        i = 0
        exceptions = ["går"]
        while i < len(doc):
            tok = doc[i]
            text = tok.text.lower()
            form = tok.morph.get("VerbForm")
            if len(form) > 0:
                form = form[0]
                #print(tok.text, ": ", form)
            else:
                form = "None"
            if tok.pos_ in ["VERB"] and form != "Part" and tok.text not in exceptions and not text[-1] in ["e", "r"]:
                new = tok.text.lower() + "e"
                if new in bokmal or new in nynorsk:
                    new_pos = self.nlp(new)[0].pos_
                    #print(new, ": ", new_pos)
                    if new_pos == "VERB":
                        yield i, i+1, "apocope"
            i += 1

####################################################################
# Labeling function for voicing of consonants between vowels or syllable final
####################################################################
class VoicingAnnotator(SpanAnnotator):
    def __init__(self, name, bokmal, nynorsk):
        super(VoicingAnnotator, self).__init__(name)
        self.bokmal = bokmal
        self.nynorsk = nynorsk
    #
    def devoice(self, word):
        voiceable_consts = {"b": "p", "g": "k", "d": "t"}
        vowels = ['a', 'e', 'i', 'o', 'u', 'æ', 'ø', 'y']
        devoiced = ''
        for i, char in enumerate(word.lower()):
            if i == 0:
                devoiced += char
            elif i == len(word) - 1:
                if char in voiceable_consts:
                    prev_char = word[i-1]
                    if prev_char in vowels:
                        devoiced += voiceable_consts[char]
                    else:
                        devoiced += char
                else:
                    devoiced += char
            elif char in voiceable_consts:
                    prev_char = word[i-1]
                    next_char = word[i+1]
                    if prev_char in vowels and next_char in vowels:
                        devoiced += voiceable_consts[char]
                    else:
                        devoiced += char
            else:
                devoiced += char
        return devoiced
    #
    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        i = 0
        exceptions = ["og", "lag", "med", "veg"]
        while i < len(doc):
            tok = doc[i]
            if tok.text.lower() not in exceptions:
                devoiced = self.devoice(tok.text)
                if (devoiced != tok.text.lower()) and ((devoiced in self.bokmal) or (devoiced in self.nynorsk)):
                    yield i, i+1, "voicing"
            i += 1

####################################################################
# vowel shift
####################################################################



####################################################################
# lexical
####################################################################


####################################################################
# dem_pro
####################################################################
def dem_pro(doc):
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.pos_ in ["PROPN"]:
            if i-1 >= 0:
                prev_tok = doc[i-1]
                if prev_tok.text.lower() in ["han", "n", "hun", "hu", "ho", "a"]:
                    yield i-1, i+1, "dem_pro"
        i += 1

####################################################################
# shortening
####################################################################
def shortening(doc):
    short = ["pottet"]
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.text.lower() in short:
            yield i-1, i+1, "shortening"
        i += 1

####################################################################
# gender
####################################################################
def gender(doc):
    gendered = ["jenten", "pausa"]
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.text.lower() in gendered:
            yield i-1, i+1, "gender"
        i += 1


####################################################################
# marked
####################################################################
class MarkedAnnotator(SpanAnnotator):
    def __init__(self, name, marked):
        super(MarkedAnnotator, self).__init__(name)
        self.marked = marked
    #
    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        i = 0
        while i < len(doc):
            tok = doc[i]
            if tok.text.lower() in self.marked:
                yield i, i+1, "marked"
            i += 1

####################################################################
# h_v
####################################################################

def h_v(doc):
    forms = ["hårr", "ka", "kem", "ke",
             "kors", "korsen", "kor",
             ]
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.text.lower() in forms:
                yield i, i+1, "h_v"
        i += 1

####################################################################
# adjectival_declension
####################################################################
"""
ekje så møje større enn ein store hond
"""
def adj_dec(doc):
    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.pos_ in ["ADJ"] and tok.text.lower().endswith("e"):
            if i + 1 < len(doc) and i-1 >= 0:
                prev_tok = doc[i-1].text.lower()
                next_tok = doc[i+1]
                next_pos = next_tok.pos_
                if prev_tok in ["en", "ein", "et", "eit"] and next_pos is "NOUN":
                    yield i, i+1, "adjective_declension"
        i += 1

####################################################################
# nominal_declension
####################################################################
def nom_dec(doc):
    i = 0
    exceptions = ["ski"]
    while i < len(doc):
        tok = doc[i]
        if tok.pos_ in ["NOUN"] and tok.text.lower().endswith("i") and tok.text.lower() not in exceptions:
            yield i, i+1, "nominal_declension"
        i += 1


####################################################################
# conjugation
####################################################################
def conjugation(doc):
    i = 0
    exceptions = ["vet", "kan", "skal", "vil", "finnes"]
    while i < len(doc):
        tok = doc[i]
        if tok.pos_ in ["VERB"] and not tok.text.lower().endswith("r") and tok.text not in exceptions:
            tense = tok.morph.get("Tense")
            if len(tense) > 0:
                tense = tense[0]
            else:
                tense = "None"
            if tense != "Past":
                if i + 1 < len(doc) and i-1 >= 0:
                        prev_tok = doc[i-1].text.lower()
                        prev_pos = doc[i-1].pos_
                        next_tok = doc[i+1]
                        next_pos = next_tok.pos_
                        if prev_tok in ["jeg", "eg", "je", "jæ", "jæi", "æ", "ej", "han", "hun", "den", "vi", "me", "de", "dere", "dokker", "dokk", "døkker", "døk", "dom"]:
                            if prev_pos not in ["AUX"] and next_pos not in ["AUX"]:
                                yield i, i+1, "conjugation"
        i += 1

####################################################################
# functional
####################################################################

class FunctionalAnnotator(SpanAnnotator):
    def __init__(self, name, functional):
        super(FunctionalAnnotator, self).__init__(name)
        self.functional = functional
    #
    def find_spans(self, doc: Doc) -> Iterable[Tuple[int, int, str]]:
        i = 0
        while i < len(doc):
            tok = doc[i]
            if tok.text.lower() in self.functional:
                yield i, i+1, "functional"
            i += 1


####################################################################
# phonemic_spelling
####################################################################


def read_dic(dic):
    vocab = set()
    for line in open(dic):
        # skip lines that are commented out
        if not line.startswith("#"):
            vocab.add(line.strip().split("/")[0].lower())
    return vocab


if __name__ == "__main__":

    import spacy
    nlp = spacy.load("nb_core_news_sm")

    labels = ["pron_subj",
              "pron_obj",
              "copulate",
              #"contraction",
              #"palatalization",
              "present_marker_deletion",
              #"apocope",
              "voicing",
              #"vowel_shift",
              #"lexical",
              #"dem_pro",
              #"shortening",
              #"gender",
              #"marked",
              "h_v",
              #"adjectival_declension",
              #"nominal_declension",
              #"conjugation",
              #"functional",
              #"phonemic_spelling",
              #"interjection"
              ]

    texts = ['det var noe av det beste jeg her sett!',
             'æ e så forbainna',
             'det går vel tilbage til ei bog eg leste',
             'dem har ikke noe å si',
             'så godt har dem aldri spilt',
             'jeg har ikke sett dem enda',
             'eg sa de i går',
             'eg vet ikkje ka som har skjedd',
             'eg vet ikke hva som har skjedd',
             "det ha'kke noe å si",
             "ekje så møje større enn ein store hond",
             "E kje så møye mindre enn ein liden hond heller",
             "Sku ITTE ha skrivi dæ . Sogndal skårer . Litt unødvendig . https : //t.co/IzQKl9iqeb",
             "Nei tore æ vil ikke snakk me dæ",
             "Vess du kun e på treningssenteret før å se stygt på andre så kan du faktisk reis tel helvette vekk derfra.",
             "åssen går det me deg?",
             "Jaja , du trega kanskje atta du inkje gjer det au ; men det æ det , he du fysst gjort det så sede du i klemmå.Då kan du jeispa , du kjeme ingjen veg",
             "Æ har møtt veggen . Den heite kjemi 2 og den e 3 lag mur og 8 lag stål og 10 lag vibranium : - )",
             "Eg såg snurten av innhaldslista til boki Anders Aschim og Per Hasle skriv til hundradårshøgtidi for Fyrebilsbibelen , og der er det ei underyverskrift « Dette med Mortensson » – ja , med hermeteikn . Og det segjer eg for sant og visst : Dette er pirresnutt som duger . https : //t.co/8L2NQnjTRr",
             "Ein av to ganger denne sesongen eg satse på @ leoskirio har ein rævva kamp https : //t.co/cOvzoHEONk",
             "Fyflate kor ej suge på å huske å gratulere folk på face ......",
             "Sommerskirenn , jida greit nok det , men rulleski er juks ! Skarru gå på ski får ' u gå på ski ! Staka på asfalten !",
             """Etter en flott kveld med ho Kari på @hteater
 blir d nedtur med mer danseshow på #tv2. Stoltenberg sr. funke hos @FredrikSkavlan"""
             ]

    docs = list(nlp.pipe(texts))

    bokmal = read_dic("bokmal.dic")
    nynorsk = read_dic("nynorsk.dic")
    functional_dict = read_dic("functional.txt")
    marked_dict = read_dic("marked.txt")

    annotator = NordialAnnotator(nlp,
                                 bokmal,
                                 nynorsk,
                                 functional_dict,
                                 marked_dict)
    annotator.add_all()

    docs = list(annotator.pipe(docs))

    hmm = HMM("hmm", labels)
    #hmm.fit_and_aggregate(docs)
