import pandas as pd
import numpy as np
import langid
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words("german"))
stemmer = SnowballStemmer("german")



import string
# string.punctuation
# string.punctuation

import re
from nltk.stem.snowball import SnowballStemmer

from gensim.models import Word2Vec
import seaborn as sns
sns.set()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from gensim.models.fasttext import FastText


from sklearn.metrics import precision_score, recall_score,f1_score
import joblib

# Load data
df = pd.read_csv('C:/Users/DELL/Desktop/Visable/sample_data.csv')
categories = list(df['label'].value_counts().index)
print("List of classes: ",categories, " & Total number of classes are: ", df['label'].value_counts().count())

# Visualize to view the categorical distribution
sns.set(rc={'figure.figsize':(12,5)})
ax = sns.countplot(df['label'])
for p in ax.patches:
        ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x() + 0.05, p.get_height() + 80))

#  Data Cleaning
print("Total number of null/missing values in the data frame :",df.isnull().values.sum())
df.drop_duplicates(keep=False, inplace=True)

def deutsch_it(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[dataframe['text'].apply(lambda row: langid.classify(row)[0] == 'de')]
    return dataframe

de_df = deutsch_it(df)
de_df.head(3)


def re_text(text):
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('#\S+', '', text)  # remove #
    text = re.sub('@\S+', '  ', text)  # remove @
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~•▪︎➦"""), ' ', text)  # remove signs and bullets
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    text = re.sub('\n', ' ', text)
    text = re.sub('Null', '', text)
    text = "".join([i for i in text if i not in string.punctuation])
    text = ''.join([i for i in text if not i.isdigit()])
    stop_words_lambda = lambda x: ' '.join([word for word in x.split() if word not in (stop_words)])
    text = stop_words_lambda(text)
    text = word_tokenize(text)
    text = [stemmer.stem(word) for word in text]
    return text

de_df['stemmed_tokens'] = de_df['text'].apply(lambda x:re_text(x))

le = LabelEncoder()
de_df['label_code'] = le.fit_transform(de_df['label'])

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

final_df = de_df[['label_code','stemmed_tokens']]


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(final_df['stemmed_tokens'],
                                                    final_df['label_code'], test_size=0.2)

w2v_model = Word2Vec(sentences=final_df['stemmed_tokens'],vector_size=300,window=10,min_count=1)
# w2v_model.train(train['stemmed_tokens'],epochs=32,total_examples=len(train['label_code']))
w2v_model.train(X_train,epochs=32,total_examples=len(y_train))

# Save the model
model_filename = 'w2v_model_final'
print("Saving model to {}...".format(model_filename))
joblib.dump(w2v_model, model_filename)

words = set(w2v_model.wv.index_to_key )
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_train])
X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_test])


# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(100, dtype=float))
        
X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype=float))


# Instantiate and fit a basic Random Forest model on top of the vectors
rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect_avg, y_train.values.ravel())
y_pred_rf = rf_model.predict(X_test_vect_avg)

print("Precision RandomForest: ",precision_score(y_test, y_pred_rf, average=None))
print("Recall RandomForest: ",recall_score(y_test, y_pred_rf, average=None))
print("f1_score RandomForest: ",f1_score(y_test, y_pred_rf, average=None))


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_model = nb.fit(X_train_vect_avg, y_train.values.ravel())
y_pred_nb = nb_model.predict(X_test_vect_avg)
print("Precision naive_bayes: ",precision_score(y_test, y_pred_nb, average=None))
print("Recall : naive_bayes",recall_score(y_test, y_pred_nb, average=None))
print("f1_score : ",f1_score(y_test, y_pred_nb, average=None))

# Clearly Random forest gives better predictions

# Save the model
model_filename = 'classifier_final'
print("Saving model to {}...".format(model_filename))
joblib.dump(rf_model, model_filename)

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)


def predictor(text,model):
    text = re_text(text)
    test_vect = np.array([np.array([w2v_model.wv[i] for i in text if i in words])] )
    X_test_vect_avg = []
    for v in test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))
    y = model.predict(X_test_vect_avg)
    return dict((v,k) for k,v in le_name_mapping.items()).get(y[0])

# tt = 'Lebensmittel kommssionierung'
# print(predictor(tt,rf_model)) 
