import os
from collections import defaultdict
from pathlib import Path

#---#---#---#---#---#---#---#---#---#---#---#---#
#               POSTPROCESSING                  #
# obs: might take some time to run the reading  #
#---#---#---#---#---#---#---#---#---#---#---#---#


alle = list(os.walk("."))
#print(len(list(os.walk("."))))

total = defaultdict(dict)

# the resulting tree: place 0 is the current directory, place 1 are other directories, place 2 are files
for directory in alle:
    for fil in directory[2]:
        if ".py" not in fil:
            # Finding common name for .txt and .ann-files 
            file_base = fil[:-4]
            # Path to add for reading
            fullpath = os.path.join(directory[0],fil)
            # (Unique) path for adding
            # This is because some tweets ending up having the same IDs
            name_path = os.path.join(directory[0],file_base)
            if ".ann" in fil:
                with open(fullpath,"r",encoding="utf-8") as ann_data:            
                    total[name_path]["annotation"] = ann_data.read()
            elif ".txt" in fil:
                with open(fullpath,"r",encoding="utf-8") as text_data:            
                    total[name_path]["text"] = text_data.read()
                

print("Total files")
print(len(total))


# Checking annotations
types = defaultdict(list)
for file_name in total:
    #print("------------{}-----------".format(file_name))
    if "annotation" in total[file_name]:
        for line in total[file_name]["annotation"].split("\n"):
            info = line.split("\t")
            if info != ['']:
                label = info[1].split(" ")[0]
                types[label].append(info[2]) #info[2] is the token



for t in types:
    outfilename = t + "output.txt"
    with open(os.path.join("outputfiles_data",outfilename),"w",encoding="utf-8") as outdata:
        for word in set([x.lower() for x in types[t]]):
            outdata.write(word+"\n")
            

    


#Sorterer og skriver ut etter frekvens
#types_liste = [(x,len(y)) for x,y in types.items()]
#types_liste.sort(key=lambda x:x[1],reverse=True)


#for t in types_liste:
#    print("{:<24}{}".format(t[0],t[1]))
    






#from pathlib import Path
#result = list(Path(".").rglob("*.[tT][xX][tT]"))
#for f in folders:
    

    

