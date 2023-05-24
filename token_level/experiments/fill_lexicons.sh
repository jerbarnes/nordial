#!/bin/bash

# Functional
grep "functional" ../Finished_wordlevel/*/*/* | cut -f 3 | tr '[:upper:]' '[:lower:]' | sort | uniq >> lexicons/functional.txt

# Marked
grep "marked" ../Finished_wordlevel/*/*/* | cut -f 3 | tr '[:upper:]' '[:lower:]' | sort | uniq >> lexicons/marked.txt

# Copula
grep "copula" ../Finished_wordlevel/*/*/* | cut -f 3 | tr '[:upper:]' '[:lower:]' | sort | uniq >> lexicons/copula.txt

# Contraction
grep "contraction" ../Finished_wordlevel/*/*/* | cut -f 3 | tr '[:upper:]' '[:lower:]' | sort | uniq >> lexicons/contraction.txt

# Present_marker_deletion
grep "present_marker_deletion" ../Finished_wordlevel/*/*/* | cut -f 3 | tr '[:upper:]' '[:lower:]' | sort | uniq >> lexicons/present_marker_deletion.txt

# shortening
grep "shortening" ../Finished_wordlevel/*/*/* | cut -f 3 | tr '[:upper:]' '[:lower:]' | sort | uniq >> lexicons/shortening.txt

# gender
grep "gender" ../Finished_wordlevel/*/*/* | cut -f 3 | tr '[:upper:]' '[:lower:]' | sort | uniq >> lexicons/gender.txt

# h_v
grep "h_v" ../Finished_wordlevel/*/*/* | cut -f 3 | tr '[:upper:]' '[:lower:]' | sort | uniq >> lexicons/h_v.txt

# phonemic_spelling
grep "phonemic_spelling" ../Finished_wordlevel/*/*/* | cut -f 3 | tr '[:upper:]' '[:lower:]' | sort | uniq >> lexicons/phonemic_spelling.txt
