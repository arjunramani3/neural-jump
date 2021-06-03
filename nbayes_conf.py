#Implement naive bayes for journalist confidence
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

    with open('/Users/arjun/Documents/cs224n/deepjump/words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())
    return valid_words

#read in and clean csv of labels
#Corporate Profits, Government Spending, Macroeconomic News & Outlook, International Trade Policy, Monetary Policy and Sovereign Military Actions
def load_labels():
    """load_labels function
        @returns labels (dataframe of slug, label) tuples"""

    labels = pd.read_csv('/Users/arjun/Documents/cs224n/deepjump/jumps_by_day.csv')
    #print(lab_map)
    cols_to_keep = ['Date', 'Return', '(#) Journalist Confidence'] #specification in paper
    labels = labels[cols_to_keep]
    labels['Date'] = pd.to_datetime(labels['Date'], errors='coerce', infer_datetime_format=True)
    labels = labels.rename(columns={'(#) Journalist Confidence': 'Confidence'})
    labels = labels.round({'Confidence': 0})
    return labels

#read in articles using import_article and labels with load_labels. Take first nwords of each article
#and create dataframe with columns Date, Words (the words in the article), [labels] where [labels] is 
#the set of labels kept from load_labels
def load_articles(narts=5, nwords = 100, min_word_length = 3, filter_stop_words = True):
    """load_articles function
    @param narts (int): the number of articles to store in the labeled dataframe
    @param nwords (int): the number of words to keep in each article
    @param min_word_length (int): the minimum number of characters in a word (all words with length < min_word_length will be filtered)
    @param filter_stop_words (boolean): a flag indicating whether to filter stop_words 
    @return labeled_articles (DataFrame): a dataframe with aritcle clippings and associated labels"""
    #print('narts = ' + str(narts))
    english_words = load_eng_words()
    stop_words = set(stopwords.words('english'))
    labels = load_labels()
    print('len(labels = ' + str(len(labels)))
    print(pd.Series(labels['Confidence']).value_counts())
    #print(labels.head())
    articles = pd.DataFrame(np.zeros((narts, 2)), columns = ['Date', 'Words'])
    i = 0
    for art in os.listdir('/Users/arjun/Documents/cs224n/deepjump/WSJ_txt'):
        #print(art)
        if i > narts: break
        rawart=import_article(art, english_words, stop_words, min_word_length, filter_stop_words)
        #print(len(rawart.split(" ")))
        #print("RAW ARTICLE: " + str(rawart))
        rawart=rawart.split(" ")
        #print('len(rawart = ' + str(len(rawart)))
        #if len(rawart) < nwords: continue
        firstn = rawart[:nwords]
        firstn = " ".join(firstn) #if our input is a text with spaces
        #print(firstn)
        slug = art.split('.')[0]
        articles.loc[i] = slug, firstn
        i+=1
    #print(labels.head())
    articles['Date'] = articles['Date'].str.replace('_', '/')
    #print(articles['Date'])
    articles['Date'] = pd.to_datetime(articles['Date'], errors='coerce', format='%Y/%m/%d') 
    print('len(articles) = ' + str(len(articles)))
    labeled_articles = labels.merge(articles, left_on = 'Date', right_on = 'Date')
    print('len(labeled_articles) = ' + str(len(labeled_articles)))
    return labeled_articles

def test(narts=5, nwords = 100, min_word_length = 3, filter_stop_words = True):
     #####Implementing Naive Bayes#####
    print('running with parameters' + str((narts, nwords, min_word_length, filter_stop_words)))
    labeled_articles = load_articles(narts, nwords, min_word_length, filter_stop_words)
    print(labeled_articles.head())
    print(len(labeled_articles))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(labeled_articles['Words'], labeled_articles['Confidence'], random_state=2018, test_size = .10)
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
    #print out predicted labels and actual labels 
    combined = pd.DataFrame()
    combined['predictions'] = list(predictions)
    combined['y_test'] = list(y_test)
    combined.to_csv('combined10.csv')
    print('combined = ' + str(combined.head()))
    #print(pd.Series(y_test).value_counts())
    #print(pd.Series(predictions).value_counts())

def main():
    #Run Naive Bayes and print output with various parameters
    # test(1104, 50, 3, True)
    # test(100, 100, 3, True)
    # test(300, 100, 3, True)
    # test(500, 100, 3, True)
    # test(700, 100, 3, True)
    # test(900, 100, 3, True)
    # test(1000, 100, 3, True)
    test(1200, 100, 3, True)
    # test(1104, 150, 3, True)
    # test(1104, 200, 3, True)
    # test(1104, 250, 3, True)
    # test(1104, 300, 3, True)
    # test(1104, 350, 3, True)
    # test(1104, 400, 3, True)
    # test(1104, 450, 3, True)
    # test(1104, 500, 3, True)

if __name__ == "__main__":
    main()
