import os
import json

# Status script

# Checking the status of the annotations,
# both old and new.


# Samia and Petter annotations

with open(os.path.join("v1.1","data","dev.json"),"r",encoding="utf-8") as data:
    dev_sp = json.load(data)
with open(os.path.join("v1.1","data","train.json"),"r",encoding="utf-8") as data:
    train_sp = json.load(data)
with open(os.path.join("v1.1","data","test.json"),"r",encoding="utf-8") as data:
    test_sp = json.load(data)

alle_sp = dev_sp + train_sp + test_sp

alle_sp_id = {}
for tweet in alle_sp:
    alle_sp_id[tweet["sent_id"]] = tweet

# Sentence level, curated round 1 and 2
with open(os.path.join("gui_annotations",
                    "finished_anns","curated_annotations",
                    "round1_curated.json"),"r",encoding="utf-8") as data:
    runde1 = json.load(data)
with open(os.path.join("gui_annotations",
                    "finished_anns","curated_annotations",
                    "round2_curated.json"),"r",encoding="utf-8") as data:
    runde2 = json.load(data)

# Currently in progress sentence level

with open(os.path.join("gui_annotations","marie","m_final_round.json"),"r",encoding="utf-8") as data:
    marie_inprogress = json.load(data)
with open(os.path.join("gui_annotations","alexandra","a_final_round.json"),"r",encoding="utf-8") as data:
    alexandra_inprogress = json.load(data)


def get_curated_num(json_file):
    # Get the number of curated sentences from the sentence
    # level annotations.
    uncorrected = 0
    corrected = 0
    for tweet in json_file:
        if json_file[tweet]["corrected_category"] == "NONE":
            uncorrected += 1
        else:
            corrected += 1
    summen = uncorrected + corrected
    assert summen == len(json_file)
    print("Corrected:",corrected)
    print("Uncorrected:",uncorrected)
    print(corrected/(summen/100),"% corrected")
        
        
# Uncomment to get the annotations
#get_curated_num(marie_inprogress)
#get_curated_num(alexandra_inprogress)

# Check overlap

#finegrained


def get_overlapping(progress):
    for tweet in progress:
        sid = progress[tweet]["sent_id"]
        if sid in alle_sp_id:
            print(sid)
get_overlapping(marie_inprogress)
get_overlapping(alexandra_inprogress)
