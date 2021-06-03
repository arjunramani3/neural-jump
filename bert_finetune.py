import torch
from transformers import BertTokenizer, DistilBertTokenizer
#import torch.optim as optim
from import_art_stop_allyrs_v2 import import_article
import os
import pandas as pd
#import torch.nn as nn
import random
import numpy as np
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

device = torch.device("cuda")

def load_eng_words():
    """load_eng_words function
        @returns valid_words (set of strings): the set of string in the english words file """

    with open('/home/deepjump/deepjump/words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())
    return valid_words

# read in and clean csv of labels
# Corporate Profits, Government Spending, Macroeconomic News & Outlook, International Trade Policy, Monetary Policy and Sovereign Military Actions


def load_labels():
    """load_labels function
        @returns labels (dataframe of slug, label) tuples"""

    labels = pd.read_csv('/home/deepjump/deepjump/jumps_by_day.csv')
    labs = ['Corporate', 'Govspend', 'Macro', 'Monetary', 'Sovmil']
    # Create dicitonary mapping labs to indices
    lab_map = {name: i for i, name in enumerate(labs)}
    # print(lab_map)
    cols_to_keep = ['Date', 'Return'] + labs  # specification in paper
    labels = labels[cols_to_keep]
    labels['Date'] = pd.to_datetime(labels['Date'], errors='coerce', infer_datetime_format=True)
    labels['Sum'] = labels[labs].sum(axis=1)
    labels = labels[labels['Sum'] > 0]
    labels['Max'] = labels[labs].idxmax(axis=1)  # Max column has label to keep
    labels['Max'] = labels['Max'].map(lab_map)
    return labels

# read in articles using import_article and labels with load_labels. Take first nwords of each article
# and create dataframe with columns Date, Words (the words in the article), [labels] where [labels] is
# the set of labels kept from load_labels

def load_articles(narts=5, nwords=100):
    """load_articles function
    @param narts (int): the number of articles to store in the labeled dataframe
    @param nwords (int): the number of words to keep in each article
    @return labeled_articles (DataFrame): a dataframe with aritcle clippings and associated labels"""
    #print('narts = ' + str(narts))
    english_words = load_eng_words()
    stop_words = None  # set(stopwords.words('english'))
    labels = load_labels()
    # print(labels.head())
    articles = pd.DataFrame(np.zeros((narts, 2)), columns=['Date', 'Words'])
    ind = 0
    for i, art in enumerate(os.listdir('/home/deepjump/deepjump/WSJ_txt')):
        # print(art)
        if i > narts:
            break
        rawart = import_article(art, english_words, stop_words)
        if len(rawart) < nwords:
            continue
        firstn = rawart[0:nwords]
        firstn = " ".join(firstn)  # if our input is a text with spaces
        slug = art.split('.')[0]
        articles.loc[ind] = slug, firstn
        ind += 1
    print(articles.tail(20))
    articles['Date'] = articles['Date'].str.replace('_', '/')
    articles['Date'] = pd.to_datetime(
        articles['Date'], errors='coerce', format='%Y/%m/%d')
    labeled_articles = labels.merge(articles, left_on='Date', right_on='Date')
    print(labeled_articles.tail(20))
    return labeled_articles

prefix_length = 300
#####Implementing Naive Bayes#####
labeled_articles = load_articles(700, prefix_length)
all_data = np.array(labeled_articles[['Words', 'Max']])
sentences = all_data[:,0]
labels = np.array(all_data[:,1], dtype=np.int64)
print('Loading BERT tokenizer...')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

input_ids = []
for sent in sentences:
    encoded_sent = tokenizer.encode(sent, add_special_tokens = True, max_length=prefix_length)#, return_tensors='pt')
    input_ids.append(encoded_sent)

attention_masks = []
for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

from sklearn.model_selection import train_test_split
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,random_state=2018, test_size=0.1)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 16
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

from transformers import DistilBertForSequenceClassification, BertForSequenceClassification, AdamW, BertConfig

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 6, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

model.cuda()

params = list(model.named_parameters())
#for name, param in params:
#    if 'classifier' not in name:
#        param.requires_grad = False

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

from transformers import get_linear_schedule_with_warmup
epochs = 200
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,num_training_steps = total_steps)


import numpy as np
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

import random
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

loss_values = []
val_acc = []
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 5 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        #print("B LABELS SHAPE: " + str(b_labels.shape))
        #print(b_labels)
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)#, token_type_ids=None)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Running Validation...")
    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    val_acc.append(eval_accuracy/nb_eval_steps)
    np.savez('distilbert', val_acc, loss_values)
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

