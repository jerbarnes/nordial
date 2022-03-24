## Introduction

These are the guidelines for the word-based part of the NorDial annotation project. We aim to mark interesting tokens and sub-tokens in tweets, based on an earlier, sentence-based round of annotation. Sentences that were classified as "dialectal", as opposed to "bokmål" or "nynorsk", will then be subject to this word-level annotation.

One thing to note is that in some cases where there is some subjective judgment involved, we want to focus on the *dialectal* impact a word has. Annotating variation in this way is difficult, and some times there are no clear lines between dialect and non-dialect.

Note that our definitions of bokmål and nynorsk are quite strict. They are defined through the two official dictionaries "bokmålsordboka" and "nynorskordboka". Any inflection, conjugation or lemma that is not present there is seen as dialectal. This includes riksmål-like formulations. However, this does not mean that nothing within these norms contribute to our understanding of a writer's dialect. In cases where there are several choices of form, some of these might be more marked than others. See the label *marked* below for more information.

## Annotation Procedure

We use Brat () to annotate our data. The task involves identifying dialectally relevant words, and assigning the correct labels to them. One word may have more than one label. Some or many words will remain unlabeled.

## Initial Annotations

The annotators will be annotating both at tweet-level and word-level, but these guidelines are for the word-level annotations. The annotators are given 60 tweets as an initial round at the tweet-level and 50 tweets at the word level. This round is annotated equally by all annotators, and serve as an introduction and a test of the guidelines. The annotators can then report any inaccuracies or questions they have, in order to update the guidelines. This round is followed by another round of 120 tweets at the tweet level and 100 tweets at the word level. which are doubly annotated. These tweets are used to calculate  an inter-annotator-agreement (IAA) score that is used to further evaluate the effects of the guidelines and the annotators' understanding of it after round 1. 

## Labels

The task is to assign the following label to words which do not conform to the norms of nynorsk or bokmål.


#### pron-subj and pron-obj

One of the most common dialectal markers in a sentence is a pronoun. Some sentences in our data have no markers except a single pronoun, and they are therefore important for our understanding of how dialects are marked. One thing that is important when it comes to pronouns, is their syntactic function in a sentence. We therefore wish to label the subject function and object (or oblique) function separately. We do not have separate markers for the dative. 

&ensp;&ensp;&ensp;&ensp;\[...\] og **dem**(pron-subj) blir aldrig ferdige \[...\] 

#### copula

The copula være/vera/vere is a common dialectal indicator. It is marked with the label *copula*. We only mark dialectally interesting, non-standard versions of the copula, such as "e" and "værra", etc.

&ensp;&ensp;&ensp;&ensp;\[...\] at de **e**(copula) rare så klare de ikkje å forstå at de faktisk **e**(copula) rare og bare starta å shittalka tebake'


#### contraction
Contractions, especially with the negation adverb "ikke/ikkje" in its many forms. The verb and the adverb are labeled separately, but both are labeled with the "contraction" label. Contractions can occur with most auxiliary verbs together with ikke/ikkje, but we also count enclitic pronouns as contractions.

&ensp;&ensp;&ensp;&ensp;\[...\]**E|KKE** JEG SOM VILLE HA HU \[...\]

&ensp;&ensp;&ensp;&ensp;E han proff so **e|kje** det noke problem.

Other phenomena, i.e. "gå e'n tur" should not be included.

#### palatalization
Palatalization is a process where consonants have an offglide at the palate (palatum). In Norwegian this happens frequently, but not exclusively, to geminated consonants such as nn, dd and ll, in the dialects that have this trait. In writing it is usually indicated by additions of j or i. 

&ensp;&ensp;&ensp;&ensp;Æ e nok **forbainna** på denne forfølginga av folk på NAV

#### present_marker_deletion
In some dialects the final -r that marks the present tense for many verbs in the present tense in both bokmål and nynorsk is dropped. Most verbs have this -r in the written norms, but there are some notable exceptions for j-verbs in nynorsk: Velja -> vel. We also use this label to indicate the dropping of -l in present tense verb forms such as skal -> ska and vil -> vi. 

#### apocope
Apocope is the loss of word-final -a or -e. It is common in certain dialects.

#### voicing
Voicing is the process by which consonants which are voiceless in some dialects become voiced. The consonants affected are the plosives p,t and k which become b, d and g, respectively.

#### vowel_shift
A multitude of vowel changes occur throughout the Norwegian dialectal landscape. At this point we have chosen to not annotate them all separately, even though the specific type of change present is important for dialect identification. We identify changes based on their difference from the written norms, even though we are aware of that this does not necessarily indicate the historical change that has happened. Both monophtongal changes such as lowering (e->æ) and dipthongization such as e->ei are all marked with the vowel_shift label. We also see cases of monophthongization such as ei->ø. One important heuristic we follow is that we do not mark vowel shift in words that are tagged with any of the pronoun labels. The reason for this is that it is especially difficult to decide what kind of changes have occured for this class.

#### lexical
The *lexical* label is used when the lemma of a word is notably marked or non-standard (not found in the official dictionaries). Loanwords are not affected by this; the word has to be a dialectal or local version of a standard word that could have been used instead. An example is the word *tue* instead of *klut*. 

#### dem_pro
In some dialects it is common to use third person pronouns as determiners in combination with proper names. These can be full forms as in "ho Kari" or "han Olav" or reduced as in "a Kari" or "n Olav". 

#### shortening
In some dialects writers indicate a change of accent to the first syllable, with accompanying vowel reduction and consonant lengthening, by writing a double consonant after the first syllable if there is originally only one, as in "pottet" instead of "potet". This is indicated by the *shortening* label. This can also happen in cases where we have combined forms, such as "harru", where the originally long "a" in "har" is reduced to a short "a", indicated by the doubling of the following consontant "r".

#### gender
There is much variation when it comes to the grammatical gender of nouns in Norwegian. Nynorsk is more rigid, and allows variation only in cases where there is traditional variation between different genders. However, in Bokmål, the feminine gender is optional. However, there is a hierarchy. The least common remnant of the feminine gender is the indefinite article "ei". Keeping the feminine definite form -a is more common, but there is also a clear tendency to see certain high-frequency words as feminine. Examples are words like "jente". "ei jente" is slightly marked towards favoring the feminine form, while "jenten" is strongly marked towards a dialect with no feminine gender. Other words such as "sky" might fall in-between. The label "gender" is used when the choice of a feminine or masculine ending is clearly dialectally marked, as in "jenten".

#### marked
This label is used for words that are part of the written languages norms, but which are still rarely used, and therefore slightly dialectally or stylistically marked. An example is the question word *åssen* "how", which is accepted in Bokmål, but still infrequent, and somewhat marked compared to "hvordan". 

#### h_v
A notable difference between Bokmål and Nynorsk is that Nynorsk has "kv" where bokmål has "hv" in many cases. In some dialects, the "v" is lost, giving only "k" or "h", as in *hårr* "hvor" or *ka* "hva"  This is marked with the "h_v" label. When you use this label, there is no need to use phonemic_spelling.

#### adjectival_declension
The *adjectival_declension* is used for cases when an adjective takes a non-standard ending. A quite common example of this is the use of the ending "-e" in non definite or non-plural environments.  


#### nominal_declension
The *nominal_declension* label is used for cases where a noun takes a non-standard declensional ending.

#### conjugation
The *conjugation* label is used for cases where a verb takes a non-standard conjugation ending, such as *skrivi* for "skrive".

#### functional
Many functional words are spelled radically differently, and it is difficult to pinpoint all the different processes that are going on. All functional words whose spellings are not in accordance with the written norms are tagged with the label *functional*. This includes all question words, determiners, prepositions and certain adverbs.

One special case to note is that in many dialects it is difficult to decide whether the empty subject in presentation sentences (det finnes..., der er...) comes from "det", which potentially could be marked as "subject", or "der" which strictly is an adverb. We choose to annotate these cases as only "functional".

#### phonemic_spelling
In cases where there is no clear dialectal variation, but it is clear that the speaker want to indicate that they are writing a more oral form, the label *fonemic spelling* is used. This is mostly for cases where a pronunciation is very close to the perceived norm of some standard, like "næi" for "nei" (no). 

#### Interjection



#### Assimilation?

Example:

{"tweetid":{"text":"",
            "tags":{(1,3):[,f,c,v]},
	    "lemmas":{(1,3):"snø",...},
	    "variety":"dialect",
	    }
}


## Summarize all sentences here with full annotations.

&ensp;&ensp;&ensp;&ensp;\[...\] at de **e**(copula) rare så klare de ikkje å forstå at de faktisk **e**(copula) rare og bare starta å shittalka **tebake**(functional)'

&ensp;&ensp;&ensp;&ensp;\[...\]**E(contraction)|KKE(contraction)** JEG SOM VILLE HA **HU**(pronominal) \[...\]

&ensp;&ensp;&ensp;&ensp;E(copula) han proff so **e(contraction)|kje(contraction)** det **noke**(functional) problem.

&ensp;&ensp;&ensp;&ensp;**Æ**(pronominal) **e**(copula) nok **forbainna**(palatalization) på denne forfølginga av folk på NAV




Notes:
functional  - med labels
passe på "d" etc, kanskje heller ikke markert som subjekt. også kanskje ikke alle formelle subjekter.
"inkje" er bare trykksterk 
generelt vanskelig å skille dataspråk fra dialektord.
kan anta at man ikke skriver d hvis man sier da-.
contraction brukt som reduksjon i ordform
kanskje litt ukonsekvent bruk av å og og
apostrofer - med eller uten
vanskelig å avgjøre 
apokope eller contraction? kan være vanskelig å 
fysste- dialektalt , men passer ikke i noen tilfeller
han gamle mannen - også dempron
dei etc ikke alle normale subjekt 
contraction  -er det alltid med to ord eller er det bare ett. "kje" er klitisk.

ikke ta med normerte pronomener - fjerne alle subjekt-jeg etc.
third person non animate make sure syntactical label is ok
everything token level
just pretend that words like "ekje" are all words at once
we treat the whole thing as one token with apostrophe inside, "e'kje" is one token
but always one token!! apostrophes separated by spaces should not be included.
contraction is ok to use in both cases.



14:24:23 From Petter Mæhlum To Everyone:
	ekkje
	e kje
14:24:27 From Petter Mæhlum To Everyone:
	e' kje
14:24:29 From Petter Mæhlum To Everyone:
	e'kje
14:49:01 From Jeremy Barnes To Everyone:
	tokens: æ har ikkje noko
14:49:28 From Jeremy Barnes To Everyone:
	tags: sub_pron: 1
14:50:09 From Jeremy Barnes To Everyone:
	tags: [[sub_pron], []]
14:50:41 From Petter Mæhlum To Everyone:
	[æ, har, ikke, noko]
14:50:50 From Petter Mæhlum To Everyone:
	[[],[],[],[]]

