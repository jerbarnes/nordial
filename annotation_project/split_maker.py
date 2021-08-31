#makesplits
import json
import csv
from collections import Counter
import random
#two files used: adjudicated petter/samia + new



#code for updating the new texts:

# This function is used to load previously annotated data.
def loaddata(filename):
    # loads a previously annotated json file from the current directory
    with open(filename,"r",encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data

curated = loaddata("test_data_curated.json")
new = loaddata("test_data_new1.json")
extra = loaddata("test_data_new_sam.json")

c_by_id = {x["sent_id"]:x for x in curated}
n_by_id = {x["sent_id"]:x for x in new}
e_by_id = {x["sent_id"]:x for x in extra}

ids_c = set([x["sent_id"] for x in curated[1:]])
ids_n = set([x["sent_id"] for x in new[1:]])
ids_e = set([x["sent_id"] for x in extra[1:]])

#all non overlapping ids
allnew = ids_c.union(ids_n,ids_e)

print("new ids",len(allnew))




#splitter etterpå
newdata = []

for sentid in allnew:
    if sentid in c_by_id:
        x = c_by_id[sentid]
    elif sentid in n_by_id:
        x = n_by_id[sentid]
    else:
        x = e_by_id[sentid]
    ordbok = {}
    ordbok["sent_id"] = x["sent_id"]
    ordbok["text"] = x["text"]
    ordbok["category"] = x["category"]
    newdata.append(ordbok)

#print(newdata[0])
random.shuffle(newdata)
#print(newdata[0])




co = Counter([x["category"] for x in newdata])

print(co.most_common())

#have to remove "annet":
newdata = [x for x in newdata if x["category"] != "annet"]


split = len(newdata)//10

print("new len",len(newdata))

train = newdata[:(split*8)]
dev = newdata[(split*8):(split*9)]
test = newdata[(split*9):]

print(len(train),len(dev),len(test))

print("Saving data to file")
with open("dialect_train.json","w",encoding="utf-8") as outfile:
    json.dump(train,outfile,indent=4,ensure_ascii=False)

with open("dialect_dev.json","w",encoding="utf-8") as outfile:
    json.dump(dev,outfile,indent=4,ensure_ascii=False)

with open("dialect_test.json","w",encoding="utf-8") as outfile:
    json.dump(test,outfile,indent=4,ensure_ascii=False)