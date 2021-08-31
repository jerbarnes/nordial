from tkinter import Tk, Label, Button, Entry, IntVar, END, W, E
import json
import re

number = -1
all_k = []

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


def getSentences():
    with open('api_annot_test.json', 'r') as infile:
        sent_dict = json.load(infile)
        non_annotated = {}
        for k in sent_dict.keys():
            if sent_dict[k]['corrected_category'] == "NONE":
                global all_k
                all_k.append(k)
                non_annotated[k] = sent_dict[k]
                non_annotated[k]["text"] = deEmojify(non_annotated[k]["text"])
    # print(non_annotated)
    return(non_annotated)

class Checktrans:

    def __init__(self, master):
        self.master = master
        master.title("Annotating Norwegian dialectal tweets")
        master.rowconfigure(1,weight=1)
        master.columnconfigure(2,weight=1)
        master.grid_propagate(0)

        self.annotations_list = getSentences()

        # Labels
        self.annotations_results = Label(master, text=' ', wraplength=1000)
        self.annotations_results.grid(row=1, column=2)

        self.label = Label(master, text="Category  ->  Tweet:")
        self.label.grid(row=0, column=0, sticky=W)

        # Buttons and Layout
        # previous
        self.previous_button = Button(master, text="Previous", command=lambda: self.callBack(number), height = 2, width = 20)
        self.master.bind("<Left>", lambda event: self.callBack(number))
        self.previous_button.grid(column=1, row=8)

        # wrong
        self.wrong_label = Label(master, text="Wrong. Correct to: ", fg='red')
        self.wrong_label.grid(column=0, row=2)
        print(number)

        # Bokmål
        self.bm_button = Button(master, text="Bokmål", fg='red', command=lambda: self.update("bm"), height = 2, width = 20)
        self.master.bind("<b>", lambda event: self.update("bm"))
        self.bm_button.grid(column=1,row=2, sticky=W)
        # Nynorsk
        self.nn_button = Button(master, text="Nynorsk", fg='red', command=lambda: self.update("nn"), height = 2, width = 20)
        self.master.bind("<n>", lambda event: self.update("nn"))
        self.nn_button.grid(column=1,row=3, sticky=W)
        # Dialectal
        self.dl_button = Button(master, text="Dialectal", fg='red', command=lambda: self.update("dl"), height = 2, width = 20)
        self.master.bind("<d>", lambda event: self.update("dl"))
        self.dl_button.grid(column=2,row=2, sticky=W)
        # Mix
        self.mx_button = Button(master, text="Mixed", fg='red', command=lambda: self.update("mx"), height = 2, width = 20)
        self.master.bind("<m>", lambda event: self.update("mx"))
        self.mx_button.grid(column=2,row=3, sticky=W)

        # correct
        self.correct_button = Button(master, text="Correct!", fg='green', command=lambda: self.update("correct"), height = 2, width = 20)
        self.correct_button.grid(column=4, row=2, sticky=E)
        self.master.bind("<space>", lambda event: self.update("correct"))

        # next
        self.next_button = Button(master, text="Next", command=lambda: self.callNext(number), height = 2, width = 20)
        self.next_button.grid(column=4, row=8)
        self.master.bind("<Right>", lambda event: self.callNext(number))
        self.corrected_label = Label(master, text="Corrected to ", fg='green')
        self.corrected_label.grid(column=0, row=4)
        self.corrected = Label(master, text='')
        self.corrected.grid(column=1, row=4, sticky=W)

        # finish
        self.finish_button = Button(master, text="Finish!", command=lambda: self.update("finished"), height = 2, width = 20)
        self.finish_button.grid(column=4, row=12)
        self.finished_label = Label(self.master, text="")
        self.finished_label.grid(row=6, column=4, sticky=W)

    def callBack(self, globalvar):
        self.corrected.configure(text='')
        global number
        number = globalvar - 1
        self.getitem(number)

    def callNext(self, globalvar):
        self.corrected.configure(text='')
        global number
        number = globalvar + 1
        self.getitem(number)

    def getitem(self, number):
        print(number)

        self.previous_button["state"] = "normal"
        self.next_button["state"] = "normal"
        global all_k
        if number <= len(all_k)-1 and not number < 0:
            self.annotations = self.annotations_list[all_k[number]]["text"]
            self.original = self.annotations_list[all_k[number]]["category"]
            self.annotations_results.configure(text=self.original+'\n--------------\n\n\n'+self.annotations, font=("Times New Roman", 14, "bold"))
        elif number < 0:
            self.annotations_results.configure(text='Reached the start of the list')
            self.previous_button["state"] = "disabled"
        else:
            self.annotations_results.configure(text='No more elements')
            self.next_button["state"] = "disabled"

    def validate(self, new_text):
        if not new_text: # the field is being cleared
            self.entered_word = ''
            return True

        try:
            self.entered_word = new_text
            return True
        except ValueError:
            return False

    def update(self, method):
        if method == "wrong":
            global number, all_k
            self.annotations_list[all_k[number]]["corrected_category"] = self.entered_word
            self.corrected.configure(text=self.annotations_list[all_k[number]]["category"]+'  ->  '+self.annotations_list[all_k[number]]["corrected_category"])

        elif method == "correct":
            self.annotations_list[all_k[number]]["corrected_category"] = ""

        elif method == "bm":
            self.annotations_list[all_k[number]]["corrected_category"] = "bokmål"
            self.corrected.configure(text=self.annotations_list[all_k[number]]["category"]+'  ->  '+self.annotations_list[all_k[number]]["corrected_category"])

        elif method == "nn":
            self.annotations_list[all_k[number]]["corrected_category"] = "nynorsk"
            self.corrected.configure(text=self.annotations_list[all_k[number]]["category"]+'  ->  '+self.annotations_list[all_k[number]]["corrected_category"])

        elif method == "dl":
            self.annotations_list[all_k[number]]["corrected_category"] = "dialectal"
            self.corrected.configure(text=self.annotations_list[all_k[number]]["category"]+'  ->  '+self.annotations_list[all_k[number]]["corrected_category"])

        elif method == "mx":
            self.annotations_list[all_k[number]]["corrected_category"] = "mixed"
            self.corrected.configure(text=self.annotations_list[all_k[number]]["category"]+'  ->  '+self.annotations_list[all_k[number]]["corrected_category"])

        elif method == "finished":
            with open('api_annot_test.json') as originalfile:
                original = json.load(originalfile)
            original.update(self.annotations_list)

            with open('api_annot_test.json', 'w') as correctedout:
                json.dump(original, correctedout, indent=4, ensure_ascii=False)
            self.finished_label.configure(text="Congrats! You are done for today :)")



def main():
    root = Tk()

    root.geometry("1500x600") #size of the GUI
    root.resizable(0, 0) #don't allow resizing in the x or y direction

    my_gui = Checktrans(root)
    root.mainloop()

if __name__ == "__main__":
    main()
