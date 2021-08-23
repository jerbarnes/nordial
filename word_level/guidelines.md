Guidelines



find tool
write guidelines
examples
first 
60 tweets as a round 1
round 2 120 tweets doubly annotated
50 100 for word level
same for both sentence and word level
first round 1 round 2 at tweet level
then do round 1 round 2 at word level
then do word level more
callibrate word level guidelines
check inter annotator agreement




#### pron-subj

pronoun subjects


#### pron-obj
#### copula

Different varitities of "være".
#### contraction
Contractions are quite common.


#### palatalization
Usually indicated by additions of j or i. 

#### r-deletion
In some dialects the final -r that marks the present tense for many verbs in the present tense in both bokmål and nynorsk is dropped. Most verbs have this -r in the written norms, but there are some notable exceptions for j-verbs in nynorsk: Velja -> vel. We also use this label to indicate the dropping of -l in present tense verb forms such as skal -> ska and vil -> vi. 

#### apocope
Apocope is the loss of word-final -a or -e. It is common in certain dialects.

#### voicing
Voicing is the process by which consonants which are voiceless in some dialects become voiced. The consonants affected are the plosives p,t and k which become b, d and g, respectively.

#### vowel_shift
A multitude of vowel changes occur throughout the Norwegian dialectal landscape. At this point we have chosen to not annotate them all separately, even though the specific type of change present is important for dialect identification. Both monophtongal changes such as lowering (e->æ) and dipthongization such as e->ei are all marked with the vowel_shift label.

#### lexical
The *lexical* label is used when the lemma of a word is notably marked or non-standard (not found in the official dictionaries). Loanwords are not affected by this; the word has to be a dialectal or local version of a standard word that could have been used instead. An example is the word *tue* instead of *klut* ( 
#### dem_pro
In some dialects it is common to use third person pronouns as determiners in combination with proper names. These can be full forms as in "ho Kari" or "han Olav" or reduced as in "a Kari" or "n Olav". 

#### shortening
In some dialects writers indicate a change of accent to the first syllable, with accompanying vowel reduction and consonant lengthening, by writing a double consonant after the first syllable if there is originally only one, as in "pottet" instead of "potet". This is indicated by the *shortening* label.

#### gender
There is much variation when it comes to the grammatical gender of nouns in Norwegian. Nynorsk is more rigid, and allows variation only in cases where there is traditional variation between different genders. However, in Bokmål, the feminine gender is optional. However, there is a hierarchy. The least common remnant of the feminine gender is the indefinite article "ei". Keeping the feminine definite form -a is more common, but there is also a clear tendency to see certain high-frequency words as feminine. Examples are words like "jente". "ei jente" is slightly marked towards favouring the feminine form, while "jenten" is strongly marked towards a dialect with no feminine gender. Other words such as "sky" might fall in-between. The label "gender" is used when the choice of a feminine or masculine ending is clearly dialectally marked, as in "jenten".

#### marked
This label is used for words that are part of the written languages norms, but which are still rarely used, and therefore slightly dialectally or stylistically marked. An example is the question word *åssen* "how", which is accepted in Bokmål, but still infrequent, and somewhat marked compared to "hvordan". 

#### h_v
A notable difference between Bokmål and Nynorsk is that Nynorsk has "kv" where bokmål has "hv" in many cases. In some dialects, the "v" is lost, giving only "k" or "h", as in *hårr* "hvor" or *ka* "hva"  This is marked with the "h_v" label.

#### declension
The *declension* label is used for cases where a noun takes a non-standard declensional ending.

#### conjugation
The *conjugation* label is used for cases where a verb takes a non-standard conjugational ending, such as *skrivi* for "skrive".
#### functional

Many functional words are spelled radically differently, and it is difficult to pinpoint all the different processes that are going on. All functional words whose spellings are not in accordance with the written norms are tagged with the label *functional*. This includes all question words, determiners, prepositions and certain adverbs.

#### fonemic_spelling
In cases where there is no clear dialectal variation, but it is clear that the speaker want to indicate that they are writing a more oral form, the label *fonemic spelling* is used. This is mostly for cases where a pronounciation is very close to the perceived norm of some standard, like "næi" for "nei" (no). 





Example:

{"tweetid":{"text":"",
            "tags":{(1,3):[,f,c,v]},
	    "lemmas":{(1,3):"snø",...},
	    "variety":"dialect",
	    }
}






