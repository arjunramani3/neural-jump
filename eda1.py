from nltk.corpus import wordnet
from random import sample, choice
from nbayes import load_eng_words, load_labels
from import_art_stop_allyrs_v2 import import_article
import pandas as pd
import math
import os
from nltk.corpus import stopwords
import numpy as np


# Takes in an article and outputs a new article with synonyms replaced for a specified number of words
# eda1 - synonym substitution
def similar_augment(article, replace_words):
    """function similar_augment creates a new article from an existing article with synonym replacement
        @param article (list of strings): a list of words specifying an article 
        @param replace_words (int): the number of words in the article to replace (must be less than len(article))
        @return new_article (list of strings): a list of words specifying the new article"""
    total_words = len(article)
    if replace_words >= total_words: replace_words = total_words #correct case where replace_words is too big
    to_replace = sample(range(total_words), replace_words)
    new_article = article
    for i in to_replace:
        word = article[i]
        syns = wordnet.synsets(word)
        if len(syns) > 0:
            synonyms = [a for synset in syns for a in synset.lemma_names()]
            new_word = choice(synonyms)
            new_article[i] = new_word
            #print(word, new_word)
    return new_article

# Go through each article and create a matched pair v2 using easy data augmentation synonym substitution.
# Store matched pair in WSJ_augment_txt directory
def write_augmented(replace_words, narts = 5, nwords = 100, min_word_length = 2, filter_stop_words = True):
    """load_articles function
        @param narts (int): the number of articles to store in the labeled dataframe
        @param nwords (int): the number of words to keep in each article
        @param replace_words (int): the number of words to replace with synonyms in each article
        @param min_word_length (int): the minimum number of characters in a word (all words with length < min_word_length will be filtered)
        @param filter_stop_words (boolean): a flag indicating whether to filter stop_words 
        @return labeled_articles (DataFrame): a dataframe with aritcle clippings and associated labels"""
    english_words = load_eng_words()
    stop_words = set(stopwords.words('english'))
    #print(labels.head())
    for i, art in enumerate(os.listdir('./WSJ_txt')):
        if i > narts: break
        rawart=import_article(art,english_words,stop_words, min_word_length, filter_stop_words)
        #print(len(rawart.split(" ")))
        firstn=rawart.split(" ")[0:nwords]
        new_art = similar_augment(firstn, replace_words)

        firstn = " ".join(firstn) #if our input is a text with spaces
        new_art = " ".join(new_art)
        #print(firstn)
        slug = art.split('.')[0]
        #firstn and new_art are the two articles to write
        text_file = open("/Users/arjun/Documents/cs224n/deepjump/WSJ_augment_txt/" + slug + "_2.txt", "w")
        text_file.write(new_art)
        text_file.close()

# Read in a set of training data with column Words and return an augmented version with similarity replacement (in the same order)
def get_augmented(X_train, replace_words = 50):
    """load_articles function
    @param X_train (dataframe): a dataframe with article clippings (Words)
    @return X_train_new: an augmented list of article clipping with twice the length of X_train"""
    new_articles= []
    for art in X_train:
        #print('art = ' + str(art))
        firstn=art.split(" ") #get first n words
        new_art = similar_augment(firstn, replace_words) #get similar article
        firstn = " ".join(firstn) #if our input is a text with spaces
        new_art = " ".join(new_art)
        new_articles.append(new_art)
    #print(labels.head())
    return new_articles


if __name__ == "__main__":
    write_augmented(replace_words = 50, narts = len(os.listdir('./WSJ_txt')), nwords = 100)

