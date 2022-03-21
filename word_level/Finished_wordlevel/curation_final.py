import os
from collections import defaultdict
#from pathlib import Path

#---#---#---#---#---#---#---#---#---#---#---#---#
#               POSTPROCESSING for curation     #
# obs: might take some time to run the reading  #
#---#---#---#---#---#---#---#---#---#---#---#---#


iaa_marie = os.listdir("Finished_Marie_round6_IAA")
iaa_alexandra = os.listdir("IAA_alexandra")

total = defaultdict(dict)
assert len(iaa_marie) == len(iaa_alexandra)


marie= defaultdict(lambda:defaultdict(list))
for f in iaa_marie:
    with open(os.path.join("Finished_Marie_round6_IAA",f),"r",encoding="utf-8") as data:
        if ".txt" in f:
            marie[f[:-4]]["text"] = data.read().split("\n")
        elif ".ann" in f:
            marie[f[:-4]]["annotation"] = data.read().split("\n")


alexandra =  defaultdict(lambda:defaultdict(list))       
for f in iaa_alexandra:
    with open(os.path.join("IAA_alexandra",f),"r",encoding="utf-8") as data:
        if ".txt" in f:
            alexandra[f[:-4]]["text"] = data.read().split("\n")
        elif ".ann" in f:
            alexandra[f[:-4]]["annotation"] = data.read().split("\n")
        

print(len(alexandra))
print(len(marie))
anno_m = defaultdict(lambda:defaultdict(list)) #filename -> span ->list of labels
anno_a = defaultdict(lambda:defaultdict(list)) #filename -> span ->list of labels
#easy lookup for words
words_m = defaultdict(lambda:defaultdict())
words_a = defaultdict(lambda:defaultdict())

for fila in marie:#have checked that they are all there
    for annom,annoa in zip(marie[fila]["annotation"],alexandra[fila]["annotation"]):
        #
        linem = annom.split("\t")
        linea = annoa.split("\t")

        if linem != [''] and linea != ['']:
            linem_split = linem[1].split(" ")
            linea_split = linea[1].split(" ")
            index_m = tuple(linem_split[1:])
            index_a = tuple(linea_split[1:])
            anno_m[fila][index_m].append(linem_split[0])
            anno_a[fila][index_a].append(linea_split[0])
            #print(linem[2])
            words_m[fila][index_m] = linem[2] #words
            words_a[fila][index_a] = linem[2] #words
            
#for f in anno_m:
#    if f not in anno_a:
#        print(f)
# Result: the same file is missing
eni = 0
ueni = 0
#enige = []
#uenige= []
#>>> len(enige)
#57
#>>> len(uenige)
#257
#{'vowel_shift', 'phonemic_spelling', 'conjugation', 'pronoun_subject', 'nominal_declension', 'copula', 'functional', 'contraction', 'adjectival_declension', 'lexical', 'pronoun_object', 'demonstrative_pronoun', 'present_marker_deletion', 'voicing'}
#>>> set(uenige)
#{'contraction', 'functional', 'lexical', 'demonstrative_pronoun', 'pronoun_object', 'conjugation', 'pronoun_subject', 'nominal_declension', 'copula', 'adjectival_declension', 'marked', 'present_marker_deletion', 'voicing', 'vowel_shift', 'phonemic_spelling', 'h_v', 'palatalization', 'shortening', 'interjection'}
ikkemed = []
medja = []
for text in anno_m:
    for man in anno_m[text]:
        for aan in  anno_a[text]:
            if aan == man: #word is same
                mset = set(anno_m[text][man])
                aset = set(anno_a[text][aan])
                medja.append(aan)
                medja.append(man)
                if mset == aset:
                    #print("enig")
                    #enige.extend(list(mset))
                    eni += 1
                else:
                    ueni += 1
                    print(words_m[text][man])
                    #print(text)
                    print("M",mset)
                    print("A",aset)
            else:
                ikkemed.append(aan)
                ikkemed.append(man)
                
        
