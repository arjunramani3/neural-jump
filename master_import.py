from import_art_stop_allyrs_v2 import import_article
import pandas as pd
import math
import os
from nltk.corpus import stopwords
firstnwords=100

#Read the dicationary of all english words
def load_words():
    with open('./words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())
    return valid_words

# Load the dictionary of english words
english_words = load_words()

# Load english stop words
stop_words = set(stopwords.words('english'))

arts = []
for fname1 in os.listdir('./WSJ_txt'):
    #print(fname1)
    arts.append(fname1)

#Import articles into docs
docs=[]
docs_intact=[]
for art in arts:
    #print("PATH IS: " + str(art))
    rawart=import_article(art,english_words,stop_words)
    #In this code, I want to take the first X words -- start with 100, can later do more
    #print(len(rawart.split(" ")))
    #print("RAW ARTICLE: ")
    #print(rawart)
    firstp=rawart.split(" ")[0:firstnwords]
    combfirst=" ".join(firstp)
    docs.append(firstp)
    docs_intact.append(combfirst)
    #print(art)

#Get unique words
wordSet = set().union(*docs)
print(len(wordSet))

f2ings that only appear in 1 document -- max_df=0.95 -- common stuff should be dealt with by the tfidf
#cv = CountVectorizer(min_df=2, max_df=0.90, lowercase=True)
cv = CountVectorizer(min_df=2, max_df=0.80, lowercase=True)

X = cv.fit_transform(docs_intact)
vocab = cv.get_feature_names()
vocab_full=list(wordSet)

x = [i for i in vocab_full if i not in vocab]
#print(x)

#Want to delete these words in each part of the corpus
for k in range(len(docs)):
    filterdoc=[i for i in docs[k] if i not in x]
    docs[k]=filterdoc

#Need to re-create the wordSet
wordSet = set().union(*docs)
print(len(wordSet))

#Create a dictionary with the unique words in each document
#Note bowA -- was the list with the words for docA
#tdict is the dictionary for the current iteration of filling out the dictionary file
dicts=[]
for i in range(0,len(arts)):
    tdict=dict.fromkeys(wordSet, 0)
    for word in docs[i]:
        tdict[word] += 1
    dicts.append(tdict)

#Each row is a document, each column is a word
#Want to find words that are not in the vocab, but in the master,
#which implies they were delete by the filter

#This is not actually the master document -- but we can construct the master document again
mdoctemp=[]
for i in range(0,len(arts)):
    mdocpart=" ".join(docs[i])
    mdoctemp.append(mdocpart)

mdoc=" ".join(mdoctemp)
mdoc=mdoc.split(" ")

wordSet2 = set(mdoc)
tdict=dict.fromkeys(wordSet2, 0)
for word in mdoc:
    tdict[word] +=1
bigdict=pd.DataFrame.from_dict(tdict, orient='index')
bigdict.to_csv('./wordcounts_f100.csv')

#Inputs -- a dictionary, and a list with words
def computeTF(wordDict, bow):
    #Create an empty dictionary
    tfDict = {}
    #Total number of words
    bowCount = len(bow)
    for word, count in wordDict.items():
        #tf is # times word appears divided by total number of words
        tfDict[word] = count/float(bowCount)
    return tfDict

#Want to create a list with the tf lists
tfs=[]
for i in range(0,len(arts)):
    ttf= computeTF(dicts[i], docs[i])
    tfs.append(ttf)


#Compute idf
#Inputs: list where each element is a document
def computeIDF(docList):
    """ computeIDF function
    @param frame1 (string): name of the article
    @param english_words (list of strings): list of english words to keep
    @param stop_words (list of string): list of stop words to filter out
    @returns article2 (list of strings): a tokenized list of words (string) 
    """
    #Don't think I need to create the empty idf
    #idfDict = {}
    #Count number of documents
    N = len(docList)
    #because it has all the documents, all dictionaries will have all words
    idfDict = dict.fromkeys(docList[0].keys(), 0)

    #Count up total frequency of each word across all documents
    #A value of 1 implies it appears in 1 document, a value of 2 implies 2 documents, etc.
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    #Compute inverse document frequency
    #as log(# documents / # documents where word appears)
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict

#Old code --
#idfs = computeIDF([wordDictA, wordDictB, wordDictC])
#with new syntax
#Need to pass in without brackets (brackets were to make it a list)
idfs = computeIDF(dicts)

#Compute tfidf as: tf * idf
def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

tfidfs=[]
for i in range(0,len(arts)):
    ttfidf=computeTFIDF(tfs[i], idfs)
    tfidfs.append(ttfidf)

#Need to figure out a way to get the titles in there
outputdf=pd.DataFrame(tfidfs)

outputdf['fname'] = pd.Series(arts, index=outputdf.index)

outputdf.to_csv('./tfidftest_f100.csv')
