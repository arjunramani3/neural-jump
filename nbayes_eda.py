#Implement naive bayes with easy data augmentation
from import_art_stop_allyrs_v2 import import_article
import pandas as pd
import math
import os
from nltk.corpus import stopwords
import numpy as np
from nbayes import load_eng_words, load_labels
import codecs

#Steps 
#2. Loop through articles
#   A. Clean
#   B. Populate dataframe with article 
#   C. Populate dataframe with label (matching on slug)
#3. Read in easy data articles and repeat
#4. Run model


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
    for i, art in enumerate(os.listdir('/Users/arjun/Documents/cs224n/deepjump/WSJ_txt')):
        #print(art)
        if i > narts: break
        rawart=import_article(art,english_words,stop_words)
        firstn=rawart.split(" ")[0:nwords]
        firstn = " ".join(firstn) #if our input is a text with spaces
        slug = art.split('.')[0]
        articles.loc[i] = slug, firstn
    #print(labels.head())
    articles['Date'] = articles['Date'].str.replace('_', '/')
    #print(articles['Date'])
    articles['Date'] = pd.to_datetime(articles['Date'], errors='coerce', format='%Y/%m/%d') 
    labeled_articles = labels.merge(articles, left_on = 'Date', right_on = 'Date')
    #print(len(labeled_articles))
    return labeled_articles

def load_eda(narts = 5, nwords = 100):
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


def main():
    #####Implementing Naive Bayes#####
    labeled_articles = load_articles(1000, 100)
    extra_articles = load_eda(1000,100)
    labeled_articles = labeled_articles.append(extra_articles)
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
