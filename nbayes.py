#Implement naive bayes
from import_art_stop_allyrs_v2 import import_article
import pandas as pd
import math
import os
from nltk.corpus import stopwords
import numpy as np

#Steps 
#1. Create class labels 
#2. Loop through articles
#   A. Clean
#   B. Populate dataframe with article 
#   C. Populate dataframe with label (matching on slug)
#3. Run model

#Read the dictionary of all english words
def load_eng_words():
    """load_eng_words function
        @returns valid_words (set of strings): the set of string in the english words file """

    with open('./words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())
    return valid_words

#read in and clean csv of labels
#Corporate Profits, Government Spending, Macroeconomic News & Outlook, International Trade Policy, Monetary Policy and Sovereign Military Actions
def load_labels():
    """load_labels function
        @returns labels (dataframe of slug, label) tuples"""

    labels = pd.read_csv('/Users/arjun/Documents/cs224n/deepjump/jumps_by_day.csv')
    labs = ['Corporate', 'Govspend', 'Macro', 'Monetary', 'Sovmil']
    lab_map = {name : i for i, name in enumerate(labs)} #Create dicitonary mapping labs to indices
    #print(lab_map)
    cols_to_keep = ['Date', 'Return'] + labs #specification in paper
    labels = labels[cols_to_keep]
    labels['Date'] = pd.to_datetime(labels['Date'], errors='coerce', infer_datetime_format=True)
    labels['Sum'] = labels[labs].sum(axis=1)
    labels = labels[labels['Sum'] > 0] #filters out all rows that do not have a label in labs
    labels['Max'] = labels[labs].idxmax(axis=1) #Max column has label to keep
    labels['Max'] = labels['Max'].map(lab_map)
    return labels

#read in articles using import_article and labels with load_labels. Take first nwords of each article
#and create dataframe with columns Date, Words (the words in the article), [labels] where [labels] is 
#the set of labels kept from load_labels
def load_articles(narts=5, nwords = 100):
    """load_articles function
    @param narts (int): the number of articles to store in the labeled dataframe
    @param nwords (int): the number of words to keep in each article
    @return labeled_articles (DataFrame): a dataframe with aritcle clippings and associated labels"""
    #print('narts = ' + str(narts))
    english_words = load_eng_words()
    stop_words = set(stopwords.words('english'))
    labels = load_labels()
    #print(labels.head())
    articles = pd.DataFrame(np.zeros((narts, 2)), columns = ['Date', 'Words'])
    for i, art in enumerate(os.listdir('./WSJ_txt')):
        #print(art)
        if i > narts: break
        rawart=import_article(art,english_words,stop_words)
        #print(len(rawart.split(" ")))
        #print("RAW ARTICLE: " + str(rawart))
        firstn=rawart.split(" ")[0:nwords]
        firstn = " ".join(firstn) #if our input is a text with spaces
        #print(firstn)
        slug = art.split('.')[0]
        articles.loc[i] = slug, firstn
    #print(labels.head())
    articles['Date'] = articles['Date'].str.replace('_', '/')
    #print(articles['Date'])
    articles['Date'] = pd.to_datetime(articles['Date'], errors='coerce', format='%Y/%m/%d') 
    labeled_articles = labels.merge(articles, left_on = 'Date', right_on = 'Date')
    print(len(labeled_articles))
    return labeled_articles

def main():
    #####Implementing Naive Bayes#####
    labeled_articles = load_articles(1000, 200)
    print(labeled_articles.head())
    print(len(labeled_articles))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(labeled_articles['Words'], labeled_articles['Max'], random_state=1)
    print('train size = ' + str(len(X_train)))
    print('test size = ' + str(len(y_test)))

    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)

    from sklearn.naive_bayes import MultinomialNB
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train_cv, y_train)
    predictions = naive_bayes.predict(X_test_cv)

    from sklearn.metrics import accuracy_score, precision_score, recall_score
    print('Accuracy score: ', accuracy_score(y_test, predictions))
    print('Recall score: ', recall_score(y_test, predictions, average = 'macro'))
    print('Precision score: ', precision_score(y_test, predictions, average = 'weighted'))
    print(pd.Series(y_test).value_counts())
    print(pd.Series(predictions).value_counts())

if __name__ == "__main__":
    main()
