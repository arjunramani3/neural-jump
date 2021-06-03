# -*- coding: utf-8 -*-
import re

# Remove terms in the headers/footers of feach article
# Marco note: something weird about the further reproduction thing, I can't figure out why I need both b/c ignore case is on
def cleaning_code(article):
  """cleaning_code function
    @param article (list of strings): a tokenized list of words (string)
    @returns article2 (list of strings): a tokenized list of words (string)"""
    
  remlist = ['Reproduced with permission of the copyright owner', 'Wall Street Journal(1889 - 1922);',
              'ProQuest Historical Newspapers: The Wall Street Journal', 'Wall Street Journal(1923 - Current file);',
              'Further reproduction prohibited without permission',
              'further reproduction prohibited without permission',
              'ProQuest Historical Newspapers:',
              'The Wall Street Journal', 'Dow Jones', 'DowJones', 'Dow-Jones', ' pg. ']
  for term in remlist:
    article = re.sub(r'%s' % re.escape(term), '', article, flags=re.IGNORECASE)

  # print(article)

  # remove periods between numbers
  article = re.sub('(?<=\d)[.,]|[.,](?=\d)', '', article)
  # to fix Mr. Ms. Mrs. Dr. U.S. Bros. Inc. Ltd. Co. f.o.b i.e->ie
  # updated 11/9/2018 to reflect spacing issue
  remlist = [' mr. ', ' ms. ', ' mrs. ', ' dr. ', ' drs. ', ' jr. ',
              ' co. ', ' ltd. ', ' inc. ', 'l.l.c.', 'l.p.', ' bros. ',
             's.a.', ' corp. ',
             ' i.e ', ' i.e. ', ' e.g. ', ' e.g ',
             ' d.c. ', ' n.j. ', ' n.h. ', ' n.m. ', ' n.c. ', ' w.v. ', ' n.d. ',
             ' r.i. ', ' s.c. ', ' s.d. ', ' n.y. ', 'u.k.', ' u.s.a. ', 'u.s.',
             'n.e.', 'n.w.', 's.e.', 's.w.',
             'm.d.', 'a.b.', 'b.s.', 'm.b.a.', 'c.v.', 'ph.d.', 'm.s.',
             'no.r.', 'a.m.', 'p.m.',
             ' rep. ', ' sen. ' 'i.t.', ' st. ', 'f.o.b', 'www.']
  for term in remlist:
    noperterm = re.sub(r'\.', '', term, flags=re.IGNORECASE)
    article = re.sub(r'%s' % re.escape(term), noperterm, article, flags=re.IGNORECASE)

  # generic three hitters, i.e. a.b.c. need to do a few times, as it only does the first
  for j in range(2):
    article = re.sub(r'\s[a-zA-Z]\.[a-zA-Z]\.[a-zA-Z]\.\s', ' ', article, flags=re.IGNORECASE)
  # two hitters
  for j in range(2):
    article = re.sub(r'\s[a-zA-Z]\.[a-zA-Z]\.\s', ' ', article, flags=re.IGNORECASE)
  # middle names -- stranded characters with a dot
  for j in range(2):
    article = re.sub(r'\s[a-zA-Z]\.\s', ' ', article, flags=re.IGNORECASE)

  # urls with https
  article = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', article, flags=re.IGNORECASE)
  article = re.sub(r'^http?:\/\/.*[\r\n]*', ' ', article, flags=re.IGNORECASE)
  # Remove all things that look like URLs that are left, or email addresses
  article = re.sub(r'http:.*?\s', '', article, flags=re.MULTILINE)
  article = re.sub(r'\s.*?\.com\s', '', article, flags=re.MULTILINE)
  article = re.sub(r'\s.*?\.gov\s', '', article, flags=re.MULTILINE)
  article = re.sub(r'\s.*?\.org\s', '', article, flags=re.MULTILINE)
  article = re.sub(r'\s.*?\.net\s', '', article, flags=re.MULTILINE)

  # Remove month abbreviations -- these help remove big tables of returns/numbers
  # Example for big table: 1988_04_06
  months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            'jan.', 'feb.', 'mar.', 'jun.', 'jul.', 'aug.', 'sep.', 'oct.', 'nov.', 'dec.',
            'sept', 'sept.']
  for month in months:
    article = re.sub(r'\b%s\b' % re.escape(month), ' ', article, flags=re.IGNORECASE)
  # fix to deal with things like 'midnovember'
  months = ['january', 'feburary', 'march', 'april', 'may', 'june', 'july', 'august',
            'september', 'october', 'november', 'december']
  for month in months:
    article = re.sub(r'%s' % re.escape(month), ' ', article, flags=re.IGNORECASE)

  # repeat for days of the week, and the corresponding abbreviations
  days = [' sun. ', ' mon. ', ' tu. ', ' tue. ', ' tues. ', ' wed. ', ' th. ',
          ' thu. ', ' thur. ', ' thurs. ', ' fri. ', ' sat. ']
  for day in days:
    article = re.sub(r'\b%s\b' % re.escape(day), ' ', article, flags=re.IGNORECASE)

  #Fixed, 11/11
  days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
  for day in days:
    article = re.sub(r'%s' % re.escape(day), ' ', article, flags=re.IGNORECASE)

  #fix the weird apostrophes

  article = re.sub(r'’', '\'', article, flags=re.IGNORECASE)
  # replace newline with space
  article = re.sub(r'\n', ' ', article, flags=re.IGNORECASE)
  # get rid of 's, as this creates apparent typos when the apostrophe is removed
  article = re.sub(r'\'s', ' ', article, flags=re.IGNORECASE)

  # There are no hyphenated words in the dictionary
  article = re.sub(r'-', ' ', article, flags=re.IGNORECASE)
  # Deal with contractions, which will not show up as words
  article = re.sub(r"I\'m", "I am", article, flags=re.IGNORECASE)
  article = re.sub(r"I\'d", "I would", article, flags=re.IGNORECASE)
  article = re.sub(r"it\'s", "it is", article, flags=re.IGNORECASE)
  article = re.sub(r"I\'ve", "I have", article, flags=re.IGNORECASE)
  article = re.sub(r"he\'s", "he is", article, flags=re.IGNORECASE)
  article = re.sub(r"I\'ll", "I will", article, flags=re.IGNORECASE)
  article = re.sub(r"he\'d", "he had", article, flags=re.IGNORECASE)
  article = re.sub(r"we\'d", "we had ", article, flags=re.IGNORECASE)
  article = re.sub(r"it\'d", "it had", article, flags=re.IGNORECASE)
  article = re.sub(r"don\'t", "do not", article, flags=re.IGNORECASE)
  article = re.sub(r"can\'t", "can not", article, flags=re.IGNORECASE)
  article = re.sub(r"we\'re", "we are", article, flags=re.IGNORECASE)
  article = re.sub(r"isn\'t", "is not", article, flags=re.IGNORECASE)
  article = re.sub(r"won\'t", "will not", article, flags=re.IGNORECASE)
  article = re.sub(r"we\'ve", "we have", article, flags=re.IGNORECASE)
  article = re.sub(r"we\'ll", "we will", article, flags=re.IGNORECASE)
  article = re.sub(r"she\'s", "she is", article, flags=re.IGNORECASE)
  article = re.sub(r"you\'d", "you had", article, flags=re.IGNORECASE)
  article = re.sub(r"let\'s", "let us", article, flags=re.IGNORECASE)
  article = re.sub(r"who\'s", "who is", article, flags=re.IGNORECASE)
  article = re.sub(r"he\'ll", "he will", article, flags=re.IGNORECASE)
  article = re.sub(r"it\'ll", "it will", article, flags=re.IGNORECASE)
  article = re.sub(r"she\'d", "she had", article, flags=re.IGNORECASE)
  article = re.sub(r"ain\'t", "is not", article, flags=re.IGNORECASE)
  article = re.sub(r"who\'d", "who had", article, flags=re.IGNORECASE)
  article = re.sub(r"that\'s", "that is", article, flags=re.IGNORECASE)
  article = re.sub(r"didn\'t", "did not ", article, flags=re.IGNORECASE)
  article = re.sub(r"you\'re", "you are", article, flags=re.IGNORECASE)
  article = re.sub(r"you\'ll", "you will", article, flags=re.IGNORECASE)
  article = re.sub(r"what\'s", "what is", article, flags=re.IGNORECASE)
  article = re.sub(r"wasn\'t", "was not", article, flags=re.IGNORECASE)
  article = re.sub(r"you\'ve", "you have", article, flags=re.IGNORECASE)
  article = re.sub(r"aren\'t", "are not", article, flags=re.IGNORECASE)
  article = re.sub(r"here\'s", "here is", article, flags=re.IGNORECASE)
  article = re.sub(r"hasn\'t", "has not", article, flags=re.IGNORECASE)
  article = re.sub(r"hadn\'t", "had not", article, flags=re.IGNORECASE)
  article = re.sub(r"they\'d", "they had", article, flags=re.IGNORECASE)
  article = re.sub(r"here\'s", "here is", article, flags=re.IGNORECASE)
  article = re.sub(r"who\'ve", "who have", article, flags=re.IGNORECASE)
  article = re.sub(r"she\'ll", "she will", article, flags=re.IGNORECASE)
  article = re.sub(r"who\'ll", "who will", article, flags=re.IGNORECASE)
  article = re.sub(r"that\'d", "that had", article, flags=re.IGNORECASE)
  article = re.sub(r"doesn\'t", "does not", article, flags=re.IGNORECASE)
  article = re.sub(r"there\'s", "there is", article, flags=re.IGNORECASE)
  article = re.sub(r"they\'re", "they are", article, flags=re.IGNORECASE)
  article = re.sub(r"world\'s", "world", article, flags=re.IGNORECASE)
  article = re.sub(r"haven\'t", "have not", article, flags=re.IGNORECASE)
  article = re.sub(r"they\'ve", "they have", article, flags=re.IGNORECASE)
  article = re.sub(r"weren\'t", "were not", article, flags=re.IGNORECASE)
  article = re.sub(r"they\'ll", "they will", article, flags=re.IGNORECASE)
  article = re.sub(r"o\'clock", "", article, flags=re.IGNORECASE)
  article = re.sub(r"mustn\'t", "must not", article, flags=re.IGNORECASE)
  article = re.sub(r"needn\'t", "need not", article, flags=re.IGNORECASE)
  article = re.sub(r"must\'ve", "must have", article, flags=re.IGNORECASE)
  article = re.sub(r"that\'ll", "that will", article, flags=re.IGNORECASE)
  article = re.sub(r"couldn\'t", "could not", article, flags=re.IGNORECASE)
  article = re.sub(r"wouldn\'t", "would not", article, flags=re.IGNORECASE)
  article = re.sub(r"could\'ve", "could have", article, flags=re.IGNORECASE)
  article = re.sub(r"would\'ve", "would have", article, flags=re.IGNORECASE)
  article = re.sub(r"there\'ll", "there will", article, flags=re.IGNORECASE)
  article = re.sub(r"shouldn\'t", "should not", article, flags=re.IGNORECASE)
  article = re.sub(r"should\'ve", "should have", article, flags=re.IGNORECASE)

  # Now that there are no 's left, can make these spaces
  article = re.sub(r'\'', ' ', article, flags=re.IGNORECASE)


  #Remove common titles and expressions
  #no words end in pg, so last one is okay
  # 'the wall street' is there for a common import mistake
  #Need to put the longest things first -- or else they get filtered out
  remlist = ['further reproduction or distribution is prohibited without permission',
             'reproduced with permission of copyright owner',
             'abreast of the market','features of the market',
             'dow jones company inc', 'dow jones company', 'dow jones',
             'pro quest',
             ' co ', ' st ',
             'historical newspaper', 'the market', 'pg ',
             'wall street journal', 'the wall street',
             'northwestern','copyright',
             'nasdaq',
             'staff reporter', 'company inc']
  #Could just add nasdaq to dictionary
  for term in remlist:
    article = re.sub(r'%s' % re.escape(term), '', article, flags=re.IGNORECASE)

  #remove all 1 letter words
  article=' '.join([w for w in article.split() if len(w) > 1])
  return article
