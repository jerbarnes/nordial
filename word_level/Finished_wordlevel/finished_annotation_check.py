import os
from collections import defaultdict
from pathlib import Path

#folders =  [x for x in os.listdir() if x.isdir()]


alle = list(os.walk("."))
#print(len(list(os.walk("."))))

total = defaultdict

# the resulting tree: place 0 is the current directory, place 1 are other directories, place 2 are files
for directory in alle[:2]:
    for fil in directory[2]:
        file_base = fil[:-4]
        print(file_base)
        if ".ann" in fil:
            print(fil)
            fullpath = os.path.join(directory[0],fil) 
            with open(fullpath,fil),"r",encoding="utf-8") as data:            
                total[fullpath, fil)][data.read()]

# Just adding ALL files
#total = defaultdict(list)



#from pathlib import Path
#result = list(Path(".").rglob("*.[tT][xX][tT]"))
#for f in folders:
    

    

