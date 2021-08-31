
## Introduction

These are the guidelines for using the sentence-level annotation tool, and for the sentence-level annotation task.

We will be annotating in three rounds:
  - Round 1: 60 tweets, triple annotations.
  - Round 2: 120 tweets, double annotations
  - Round 3: single annotations.

We will let you know when you can move to the next round. Please do not start annotating before we let you know.

We have created a folder for each of you, containing everything you need to annotate.

## Annotation tool

To use the annotation tool, please clone the content of the repository, and then from the command line launch *gui_sentence_annotations_2.0.py* followed by the name (path) of the file you are going to annotate. For example:

```
python3 gui_sentence_annotations_2.0.py test.json
```

The interface of the annotation tool is quite simple. When you launch it, it shows nothing except a set of buttons. To start with, you have the two buttons **Previous** and **Next** that will help you navigate from one sentence to another.

To start annotating the first sentence, you have to push **Next**. The sentences are presented as:


```
category
--------------

Sentence to annotate
```


The *category* represents our four classes: *Bokmål*, *Nynorsk*, *Dialectal*, and *Mix* (a mix of words that can be of any other category). If you agree with the category assigned to the sentence, click on the **Correct** button. If you believe that the category is wrong, please provide the right answer by clicking on the button corresponding to the correct category, *i.e.* **Bokmål**, **Nynorsk**, **Dialectal**, or **Mixed**.
When you are done annotating for the day, click **Finish!**.

Some **very** important points to remember:
  - If you do not click on **Correct** the sentence will not be annotated (going to the next sentence will not mark the current one as correctly annotated).
  - If you do not click on **Finish!** your annotations will not be saved.
  - Your corrections will not be shown on the tool if you go back using **Previous** or **Next**.

We also provide keyboard shortcuts for the GUI:
  - *Left arrow* == **Previous**.
  - *Right arrow* == **Next**.
  - *Space* == **Correct**.
  - *b* == **Bokmål**.
  - *n* == **Nynorsk**.
  - *d* == **Dialectal**.
  - *m* == **Mixed**.

There are no shortcuts for the **Finish!** button.

This is a very basic tool, made to ease the annotation process. If you meet any issues when using it, please let me (@Samia) know.
