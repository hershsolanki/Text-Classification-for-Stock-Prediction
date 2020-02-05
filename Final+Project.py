#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import math
import os
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier




# Get the filenames/paths that we'll be reading
filenames = []
filepaths = []
directory = os.fsencode('./transcripts/')
for fil in os.listdir(directory):
    ticker = os.fsdecode(fil)
    if ticker != '.DS_Store':
        for f in os.listdir(os.fsencode('./transcripts/' + ticker)):
            filename = os.fsdecode(f)
            path = './transcripts/'+ticker+'/'+filename            
            filepaths.append(path);
            filenames.append(filename)


filepaths


# In[3]:



 
def extract_text_from_pdf(pdf_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
 
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
 
        text = fake_file_handle.getvalue()
 
    # close open handles
    converter.close()
    fake_file_handle.close()
 
    if text:
        return text

def extract_from_folder(filepaths):
    documents=[]
    print(len(filepaths))
    i = 0
    for path in filepaths:
        file = extract_text_from_pdf(path)
        documents.append(file)
        i += 1
        print(i)


    return documents


# In[ ]:





# In[ ]:


df = pd.DataFrame({'filename': filenames, 'content': content })
df


# In[ ]:


# Get the price data files. These files contain the over/under reaction labels that we are using to train
price_filenames = [];
price_filepaths = []

directory = os.fsencode('./prices/')
for fil in os.listdir(directory):
    ticker = os.fsdecode(fil)
    if ticker != '.DS_Store':
        filename = os.fsdecode(ticker)
        path = './prices/'+ticker        
        price_filepaths.append(path);
        price_filenames.append(filename)
        
price_filenames[0].split(' ')[0]


# In[16]:


# compile the prices into a single dateframe
mergeddf = pd.DataFrame()
for p in range(len(price_filepaths)):
    dfprices = pd.read_csv(price_filepaths[p])
    dfprices['date'] = pd.to_datetime(dfprices['date'])
    dfprices['filename'] = dfprices['date'].apply(lambda x: x.strftime('%Y%m%d') + ' - ' + price_filenames[p].split(' ')[0])
    new_order = [6, 0, 1, 2, 3, 4, 5]
    dfprices = dfprices[dfprices.columns[new_order]]

    df['filename'] = df['filename'].apply(lambda x: x.split(".")[0])

    dfprices['label'] = (dfprices['close'].shift() - dfprices['open'].shift()) > 0
    # dfprices
    merge = df.merge(dfprices, how='inner', on='filename')
    merge['label'] = merge['label'].apply(lambda x: 0 if x == False else 1)
    mergeddf = mergeddf.append(merge)


# In[ ]:





# In[17]:


def pdf_to_csv(x):
    x = x[:-3]
    return x + 'txt'


# In[2]:


X = mergeddf['content'].values
y = np.array(mergeddf['label'].tolist())


# In[24]:



# count vectorizer tests
# 5-cross-fold validation tests using multiple classifiers with Term Frequency vectors 
# (Here is Multinomial Naive Bayes).
kf = KFold(n_splits=5)
kf.get_n_splits(X)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

for i in range(3):
    for j in range(5):
        for k in range(2):
            for l in range(10):
                accuracy = 0
                recall = 0
                precision = 0
               
                for train_index, test_index in kf.split(X):
#                     print("TRAIN:", train_index, "TEST:", test_index)
                    X_train_vector, X_test_vector = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    vectorizer = CountVectorizer(input='content',ngram_range=(1,i+1), max_features=5000 + j*5000,stop_words=(None if k==1 else 'english'))
                    X_train = vectorizer.fit_transform(X_train_vector)
                    vectorizer = CountVectorizer(input='content',ngram_range=(1,i+1), vocabulary=vectorizer.get_feature_names(), max_features=5000 + j*5000,stop_words=(None if k==1 else 'english'))
                    X_test = vectorizer.fit_transform(X_test_vector)

                    clf = MultinomialNB(alpha=0.01 + l*0.01)
                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_test)
                    accuracy += sklearn.metrics.accuracy_score(pred,y_test)/5
                    recall += sklearn.metrics.recall_score(y_test,pred, average='weighted')/5
                    precision += sklearn.metrics.precision_score(y_test,pred,average='weighted')/5

                print(i,j,k,l,accuracy,recall,precision)

    


# In[62]:


# Term Frequency vs TF-IDF tests for multiple classifiers including AdaBoost, Random Forests, 
# Gaussian Naive Bayes
alphas = []
accs = []
accs2 = []
for l in range(1):
    accuracy = 0
    accuracy_2 = 0
    accuracy_svm = 0
    accuracy_svm_2 = 0
    accuracy_log = 0
    accuracy_log_2 = 0
    
    
    for train_index, test_index in kf.split(X):
    #                     print("TRAIN:", train_index, "TEST:", test_index)
        i = 1
        j = 4
        k = 0
        l = 2
        X_train_vector, X_test_vector = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        vectorizer = CountVectorizer(input='content',ngram_range=(1,i+1), max_features=5000 + j*5000,stop_words=(None if k==1 else 'english'))
        X_train = vectorizer.fit_transform(X_train_vector)
        vectorizer = CountVectorizer(input='content',ngram_range=(1,i+1), vocabulary=vectorizer.get_feature_names(), max_features=5000 + j*5000,stop_words=(None if k==1 else 'english'))
        X_test = vectorizer.fit_transform(X_test_vector)

        clf = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        accuracy += sklearn.metrics.accuracy_score(pred,y_test)/5
        
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        accuracy_svm += sklearn.metrics.accuracy_score(pred,y_test)/5
        
        clf = GaussianNB()
        clf.fit(X_train.toarray(), y_train)
        pred = clf.predict(X_test.toarray())
        accuracy_log += sklearn.metrics.accuracy_score(pred,y_test)/5
        
        print('tfidf')
        
        vectorizer = TfidfVectorizer(input='content',ngram_range=(1,i+1), max_features=5000 + 5000*j,stop_words=(None if k==1 else 'english'))
        X_train = vectorizer.fit_transform(X_train_vector)
        vectorizer = TfidfVectorizer(input='content',ngram_range=(1,i+1), vocabulary=vectorizer.get_feature_names(), max_features=5000+5000*j, stop_words=(None if k==1 else 'english'))
        X_test = vectorizer.fit_transform(X_test_vector)

        clf = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
        clf.fit(X_train.toarray(), np.array(y_train).flatten())
        pred = clf.predict(X_test.toarray());
        accuracy_2 += sklearn.metrics.accuracy_score(pred,y_test)/5
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        accuracy_svm_2 += sklearn.metrics.accuracy_score(pred,y_test)/5
        
        clf = GaussianNB()
        clf.fit(X_train.toarray(), y_train)
        pred = clf.predict(X_test.toarray())
        accuracy_log_2 += sklearn.metrics.accuracy_score(pred,y_test)/5
        
        
    alphas.append(0.01 + l*0.01)
    accs.append(accuracy)
    accs2.append(accuracy_2)
        
    print(i,j,k,l,accuracy,accuracy_2)


# In[63]:


# plt.plot(alphas,accs,'r-o', label='TF')
# plt.plot(alphas,accs2,'b-o', label='TF-IDF')
# plt.title('Accuracies for Alpha: TF vs. TF-IDF')
# plt.legend()

accuracy,accuracy_2,accuracy_svm,accuracy_svm_2,accuracy_log,accuracy_log_2
    
# plt.plot(n,pageRankScores26,'b-o',label='pageRankScores26')


# In[ ]:


plt.plot(alphas,accs2,'b-o', label='TF-IDF')


# In[4]:


# Some other TF IDF Tests, SVMs
for i in range(4):
    for j in range(2):
        for k in range(5):
            for l in range(1):
                accuracy = 0
                recall = 0
                precision = 0
                for train_index, test_index in kf.split(X):
#                     print("TRAIN:", train_index, "TEST:", test_index)
                    X_train_vector, X_test_vector = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    vectorizer = TfidfVectorizer(input='content',ngram_range=(1,i+1), max_features=5000 + 5000*k,stop_words=(None if j==1 else 'english'))
                    X_train = vectorizer.fit_transform(X_train_vector)
                    vectorizer = TfidfVectorizer(input='content',ngram_range=(1,i+1), vocabulary=vectorizer.get_feature_names(), max_features=5000+5000*k,stop_words=(None if j==1 else 'english'))
                    X_test = vectorizer.fit_transform(X_test_vector)
                    
                    clf = svm.SVC(C=1+l*10,kernel=cosine_similarity, gamma="auto")
                    clf.fit(X_train.toarray(), np.array(y_train).flatten())
                    pred = clf.predict(X_test.toarray());
                    accuracy += sklearn.metrics.accuracy_score(pred,y_test)/5
                    recall += sklearn.metrics.recall_score(y_test,pred, average='weighted')/5
                    precision += sklearn.metrics.precision_score(y_test,pred,average='weighted')/5
    
#                 print(i,j,k,l,accuracy,recall,precision)
                


# In[ ]:


# The results of the project can be seen in the Final Report. 

