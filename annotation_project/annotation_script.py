import json
import csv
#testing annotation



# Program for annotating tweets
# Opens a file. For specifying another file, please edit the file
# name "raw_data.csv". 


#Tweets are saved together with their metadata and annotations in a json-file

# Annotations are done in the terminal. Simply run the program to start.
# Write "mix" for mixed data
# Write "bok" or "bm" for bokmål data
# Write "nyn" or "nn" for nynorsk data
# Write "dia" or "d" for dialectal data
# Write "annet" for other data, or data that is problematic.
# The program saves after each annotation.
# Write "quit" to save and exit the program. 

#Example of output
#    {
#        "sent_id": "1334908482345119748",
#        "username": "Tr\u00f8nder-Avisa",
#        "date_time": "2020-12-04 17:12:38",
#        "text": "(+) \u2013 Kj\u00e6m det fleir s\u00e5nne aria no s\u00e5 g\u00e5r \u00e6! https://t.co/t5v77f0rtJ",
#        "category": "dia"
#    },



#this list holds all the data
newdata = []

# This function is used to load previously annotated data.
def loaddata(filename):
    # loads a previously annotated json file from the current directory
    with open(filename,"r",encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data

##########################################################################################
#      !!! if an annotation file exists, call the following function call: !!!           #
##########################################################################################

newdata = loaddata("test_data_new1.json")
newdata2 = loaddata("test_data_petter.json")

both = newdata + newdata2
#collects all IDs so we can automatically know what has been annotated before
ids_in_data = [tweet["sent_id"] for tweet in both]
#print(ids_in_data)
print("### {} tweets have been annotated so far ###".format(len(ids_in_data)))


#dictionary for converting the short and convenient codes (dia,bm, nn, etc.) to their long versions:
short_to_pretty = {"mix":"mixed",
    "dia":"dialectal",
    "bok":"bokmål",
    "nyn":"nynorsk",
    "annet":"annet",
    "d":"dialectal",
    "bm":"bokmål",
    "nn":"nynorsk"}


#the annotation part itself
print("\n-------------------------------------------")
print("-----# WELCOME TO THE ANNOTAION TOOL #-----")
print("-------------------------------------------\n")



with open("dialect.csv","r",encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=",")
    for line in csv_reader: #for each line (ie tweeet) in the raw_data
        # if the sentence has not been annotated before:
        if line[0] not in ids_in_data:
            ID = line[0]   #the id field of the tweet
            USER = line[1] #the username of the tweeter
            DATE = line[2] #the date
            TEXT = line[3] #the actual text, 
            inputten = input("\t\t---Hvilken kategori er dette?---\n{}\n".format(line[3])).lower()
            while inputten not in ["mix","dia","bok","nyn","annet","save","quit","bm","nn","d"]:
                inputten = input("Noe var galt. Skriv igjen.").lower()
            if inputten == "quit":
                    break
            else:
                datasett = {"sent_id":ID,
                            "username":USER,
                            "date_time":DATE,
                            "text":TEXT,
                            "category":short_to_pretty[inputten]}
                newdata.append(datasett)
                with open("test_data_new1.json","w",encoding="utf-8") as outfile:
                    json.dump(newdata,outfile,indent=4,ensure_ascii=False)
            print("\n\t\t----------ny setning----------")

print("Saving data to file")
with open("test_data_new1.json","w",encoding="utf-8") as outfile:
    json.dump(newdata,outfile,indent=4,ensure_ascii=False)
