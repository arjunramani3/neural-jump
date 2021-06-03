#Implement naive bayes with easy data augmentation
from import_art_stop_allyrs_v2 import import_article
import pandas as pd
import math
import os
from nltk.corpus import stopwords
import numpy as np
from nbayes import load_eng_words, load_labels, load_articles
import codecs
from eda1 import get_augmented

#Steps 
#2. Loop through articles
#   A. Clean
#   B. Populate dataframe with article 
#   C. Populate dataframe with label (matching on slug)
#3. Read in easy data articles and repeat
#4. Run model

def load_eda(narts = 5, nwords = 100, min_word_length = 2, filter_stop_words = True):
    """load_eda function
    @param narts (int): the number of articles to store in the labeled dataframe
    @param nwords (int): the number of words to keep in each article
    @param min_word_length (int): the minimum number of characters in a word (all words with length < min_word_length will be filtered)
    @param filter_stop_words (boolean): a flag indicating whether to filter stop_words 
    @return labeled_articles (DataFrame): a dataframe with artcle clippings and associated labels"""
    labels = load_labels()
    articles = pd.DataFrame(np.zeros((narts, 2)), columns = ['Date', 'Words'])
    for i, art in enumerate(os.listdir('/Users/arjun/Documents/cs224n/deepjump/WSJ_augment_txt')):
        if i > narts: break
        path = '/Users/arjun/Documents/cs224n/deepjump/WSJ_augment_txt/' + art
        with codecs.open(path, 'r', encoding='utf8', errors='replace') as myfile:
            rawart = myfile.read()
        firstn=rawart.split(" ")[0:nwords]
        firstn = " ".join(firstn) #if our input is a text with spaces
        slug = art.split('.')[0]
        articles.loc[i] = slug, firstn
    articles['Date'] = articles['Date'].str[:-2] #chop off _2
    articles['Date'] = articles['Date'].str.replace('_', '/')
    #print(articles['Date'])
    articles['Date'] = pd.to_datetime(articles['Date'], errors='coerce', format='%Y/%m/%d') 
    labeled_articles = labels.merge(articles, left_on = 'Date', right_on = 'Date')
    return labeled_articles


def test(narts = 5, nwords = 100, min_word_length = 2, filter_stop_words = True, replace_words = 50):
    #####Implementing Naive Bayes w/EDA#####
    print('running with parameters' + str((narts, nwords, min_word_length, filter_stop_words, replace_words)))
    labeled_articles = load_articles(narts, nwords, min_word_length, filter_stop_words) #baseline spec is (1000,100)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(labeled_articles['Words'], labeled_articles['Max'], random_state=2018, test_size = .15)

    X_train = list(X_train)
    print('old train size = ' + str(len(X_train)))
    extra_articles = get_augmented(X_train, replace_words) #extra_articles returns a list of new articles
    X_train = X_train + extra_articles #concatenate the old and new articles
    y_train = list(y_train) + list(y_train) #apply the same labels to the old and new versions
    print('new train size = ' + str(len(X_train)))

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
    # print(pd.Series(y_test).value_counts())
    # print(pd.Series(predictions).value_counts())

if __name__ == "__main__":
    test(1104, 100, 3, True, 0)
    test(1104, 100, 3, True, 10)
    test(1104, 100, 3, True, 25)
    test(1104, 100, 3, True, 40)

