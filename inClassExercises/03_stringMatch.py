# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:23:28 2020

@author: jerem
"""


# Examples for string match using RE and fuzzy matching
# This is a great way to search documents for particular words or word patterns
# fuzzy matching allows for partial matches which can help mitigate misspellings
# etc. 




import re

#https://docs.python.org/3/library/re.html
# Examples Using match and search in RE

textData = "Fuzzy Wuzzy was a bear \
Fuzzy Wuzzy had no hair \
Fuzzy Wuzzy wasn't fuzzy \
No, by gosh, he wasn't, was he? \
Silly Willy was a worm\
Silly Willy wouldn't squirm\
Silly Willy wasn't silly\
No, by gosh, he wasn't really\
Iddy Biddy was a mouse\
Iddy Biddy had no spouse\
Iddy Biddy wasn't pretty\
Oh, by gosh, it was a pity\
Fuzzy Wuzzy was a bear\
Fuzzy Wuzzy had no hair\
Fuzzy Wuzzy wasn't fuzzy\
No, by god, he wasn't, was he?"

print(re.match("c", "abcdefg"))    # No match

#%%
print(re.search("c", "abcdefg"))   # Match

#%%

print(re.match("c", "abcdefg"))    # No match
#%%


print(re.search("^c", "abcdef"))  # No match
#%%


print(re.search("^a", "abcdef"))  # Match
#%%

m = re.match(r"(\w+) (\w+)", "Isaac Newton, physicist")
print(m.group(0))       # The entire match
#'Isaac Newton'

print(m.group(1))       # The first parenthesized subgroup.
#'Isaac'

print(m.group(2))       # The second parenthesized subgroup.
#'Newton'

print(m.group(1, 2))    # Multiple arguments give us a tuple.
#('Isaac', 'Newton')

#%%
print(re.match("fuzzy", textData)) 
print(re.search("fuzzy", textData))
m= re.search("fuzzy", textData)
print(m.group(0)) 

#%%
print(re.match("Fuzzy", textData)) 
print(re.search("Fuzzy", textData))
m= re.search("Fuzzy", textData)
print(m.group(0)) 

#%%

########################################################
############### In class exercise ######################
########################################################

# Find the number of occurrences of "fuzzy" in the text data, case independent
# Find the number of 3 letter words text data
# Find the number of capitalized words in text data
# Find the indices for all of the occurrences of the word "was".


#%%
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process 

# Some Examples from geeksforgeeks


fuzz.ratio('geeksforgeeks', 'geeksgeeks') 
#87

# Exact match 
fuzz.ratio('GeeksforGeeks', 'GeeksforGeeks') 
#100

fuzz.ratio('geeks for geeks', 'Geeks For Geeks ') 
#80

# Token Sort Ratio 
fuzz.token_sort_ratio("geeks for geeks", "for geeks geeks") 
#100

# This gives 100 as every word is same, irrespective of the position 

# Token Set Ratio 
fuzz.token_sort_ratio("geeks for geeks", "geeks for for geeks") 
#88


fuzz.token_set_ratio("geeks for geeks", "geeks for for geeks") 
#100
# Score comes 100 in second case because token_set_ratio  considers duplicate words as a single word. 



# Python code showing all the ratios together, 
# make sure you have installed fuzzywuzzy module 


s1 = "I love GeeksforGeeks"
s2 = "I am loving GeeksforGeeks"
print ("FuzzyWuzzy Ratio: ", fuzz.ratio(s1, s2) )
print ("FuzzyWuzzy PartialRatio: ", fuzz.partial_ratio(s1, s2) )
print ("FuzzyWuzzy TokenSortRatio: ", fuzz.token_sort_ratio(s1, s2) )
print ("FuzzyWuzzy TokenSetRatio: ", fuzz.token_set_ratio(s1, s2) )
print ("FuzzyWuzzy WRatio: ", fuzz.WRatio(s1, s2),'\n\n')

# for process library, 
query = 'geeks for geeks'
choices = ['geek for geek', 'geek geek', 'g. for geeks'] 
print ("List of ratios: ")
print (process.extract(query, choices), '\n')
print ("Best among the above list: ",process.extractOne(query, choices) )


########################################################
############### In class exercise ######################
########################################################

# Find the number of occurrences of "fuzzy" in the text data, case independent
# Find the number of 3 letter words text data
# Find the number of capitalized words in text data
# Find the indices for all of the occurrences of the word "was".

