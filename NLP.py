# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import random

# Natural Language Tool kit
import nltk
# nltk.download('package_name') - for installation of nltk packages

# plt.style.use("ggplot")


#? 1. TOKENIZATION:

print("---------------------------------------------------------------------------------------------------------------------------------------\n")
print("1. TOKENIZATION")
print("---------------------------------------------------------------------------------------------------------------------------------------\n")


nltk.download('punkt')
testText_01 = 'I bought these for my husband and he said they are the best energy shots out there. He takes one in the mornings and works hard all day. Good stuff!'


from nltk.tokenize import word_tokenize
tokens = word_tokenize(testText_01)
print(tokens)


#? 2. STOP WORD EXCLUSION:
print("---------------------------------------------------------------------------------------------------------------------------------------\n")
print("2. STOP WORD EXCLUSION")
print("---------------------------------------------------------------------------------------------------------------------------------------\n")

nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
# print(stop_words)

filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
print("Previous: \n", tokens)
print("\nFiltered: \n", filtered_tokens)


#? 3. STEMMING / LEMMATIZATION:
print("---------------------------------------------------------------------------------------------------------------------------------------\n")
print("3. STEMMING / LEMMATIZATION")
print("---------------------------------------------------------------------------------------------------------------------------------------\n")

# Package download
nltk.download('wordnet')
# nltk.download('omw-1.4')

from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer
print(tokens)

# Stemming:

porter_stemmer = PorterStemmer()
stemmed_tokens_porter = [porter_stemmer.stem(token) for token in tokens]
print('Porter Stemming:\n \t',stemmed_tokens_porter)

snowball_stemmer = SnowballStemmer('english')
stemmed_tokens_snowball = [snowball_stemmer.stem(token) for token in tokens]
print('\n\nSnowball Stemming:\n \t', stemmed_tokens_snowball)

# Lemmatization:
# lemmatizer = WordNetLemmatizer()
# lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

#print('\n\nLemmatization:\n \t', lemmatized_tokens)


#? 4. POS Tagging:
print("---------------------------------------------------------------------------------------------------------------------------------------\n")
print("4. POS Tagging")
print("---------------------------------------------------------------------------------------------------------------------------------------\n")

nltk.download('averaged_perceptron_tagger')
taggings = nltk.pos_tag(tokens)
print(taggings)
# taggings[:5]

#? 5. Chunking:
print("---------------------------------------------------------------------------------------------------------------------------------------\n")
print("5. Chunking")
print("---------------------------------------------------------------------------------------------------------------------------------------\n")

nltk.download('words')
chunk_parser = nltk.RegexpParser(r"""
    NP: {<DT>?<JJ>*<NN>} # chunk determiner/adj+noun
    PP: {<IN><NP>} # chunk preposition+NP
    VP: {<VB.*><NP|PP|CLAUSE>+$} # chunk verbs and their arguments
    CLAUSE: {<NP><VP>} # chunk NP, VP
""")

# groups:
chunks = chunk_parser.parse(taggings)
#print(chunks)
chunks.pprint()


#? 6. NER (Named Entity Recognition):
print("---------------------------------------------------------------------------------------------------------------------------------------\n")
print("6. NER (Named Entity Recognition)")
print("---------------------------------------------------------------------------------------------------------------------------------------\n")

nltk.download('maxent_ne_chunker')
entities = nltk.ne_chunk(taggings)
entities.pprint()


#--------------------------------------------------------------------------------
#? SENTIMENTAL ANALYSIS #########################################################
#--------------------------------------------------------------------------------
print("---------------------------------------------------------------------------------------------------------------------------------------\n")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>     SENTIMENTAL ANALYSIS     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("---------------------------------------------------------------------------------------------------------------------------------------\n")

# VADER - Package Installation
nltk.download('vader_lexicon')


# Importing model
from nltk.sentiment import SentimentIntensityAnalyzer
print('Test Text:\n\t', testText_01)

# Loading model (Already Trained)
analyzer = SentimentIntensityAnalyzer()

# Measuring Scores
scores = analyzer.polarity_scores(testText_01)

print("\nScores:\n",scores)