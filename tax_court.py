import csv
import pandas as pd
import numpy as np
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#Prep
df = pd.read_excel(r'D:\Blag - DATA\tax_court_clean.xlsx', engine='openpyxl', sheet_name='Sheet1',header=0, converters={'no_putusan':str,'jenis_pajak':str,'sengketa':str,'djp_arg':str,'wp_arg':str,'pdpt_majelis':str})
df = df.apply(lambda x: x.astype(str).str.lower())

def filtering(clean):
    clean = re.sub('[\W_]+', ' ', clean)
    clean = re.sub('\d+', ' ', clean)
    clean = re.sub('\s+', ' ', clean)
    clean = re.sub('\b[a-zA-Z]\b',' ', clean)
    clean = re.sub(' +', ' ',clean)
    return clean

df['sengketa'] = df['sengketa'].apply(filtering)
df['djp_arg'] = df['djp_arg'].apply(filtering)
df['wp_arg'] = df['wp_arg'].apply(filtering)
df['pdpt_majelis'] = df['pdpt_majelis'].apply(filtering)

#Remove stopwords

factory_sw = StopWordRemoverFactory()
stopwords = factory_sw.get_stop_words()

new_stop = []
with open('D:\Blag - DATA\swindo.txt') as inputfile:
    for row in csv.reader(inputfile):
        new_stop.append(row[0])


sw_combine = factory_sw.get_stop_words()+new_stop
dictionary = ArrayDictionary(sw_combine)
stop = StopWordRemover(dictionary)

df['sengketa'] = df['sengketa'].apply(lambda x: stop.remove(x))
df['djp_arg'] = df['djp_arg'].apply(lambda x: stop.remove(x))
df['wp_arg'] = df['wp_arg'].apply(lambda x: stop.remove(x))
df['pdpt_majelis'] = df['pdpt_majelis'].apply(lambda x: stop.remove(x))
df['sengketa'] = df['sengketa'].apply(lambda x: stop.remove(x))
df['djp_arg'] = df['djp_arg'].apply(lambda x: stop.remove(x))
df['wp_arg'] = df['wp_arg'].apply(lambda x: stop.remove(x))
df['pdpt_majelis'] = df['pdpt_majelis'].apply(lambda x: stop.remove(x))

#lemmatizing or stemming
factory_st = StemmerFactory()
stemmer = factory_st.create_stemmer()

df['sengketa'] = df['sengketa'].apply(lambda x: stemmer.stem(x))
df['djp_arg'] = df['djp_arg'].apply(lambda x: stemmer.stem(x))
df['wp_arg'] = df['wp_arg'].apply(lambda x: stemmer.stem(x))
df['pdpt_majelis'] = df['pdpt_majelis'].apply(lambda x: stemmer.stem(x))

features_djp = df['djp_arg'].values
features_wp = df['wp_arg'].values
features_maj = df['pdpt_majelis'].values

labels = df.iloc[:, 7].values

vectorizer = TfidfVectorizer (lowercase=False, max_features=2500, min_df=7, max_df=0.8)
processed_features_djp = vectorizer.fit_transform(features_djp).toarray()
processed_features_wp = vectorizer.fit_transform(features_wp).toarray()
processed_features_maj = vectorizer.fit_transform(features_maj).toarray()

X_train1, X_test1, y_train1, y_test1 = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train1, y_train1)
predictions = text_classifier.predict(X_test1)
print(confusion_matrix(y_test1,predictions))
print(classification_report(y_test1,predictions))
print(accuracy_score(y_test1, predictions))

X_train2, X_test2, y_train2, y_test2 = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train2, y_train2)
y_pred2 = svclassifier.predict(X_test2)

print(f1_score(y_test2, y_pred2, average='weighted', zero_division=1))

#tokenizing
def word_tokenize_wrapper(text):
    return word_tokenize(text)

df['djp_arg2'] = df['djp_arg'].apply(word_tokenize_wrapper)

print(df['djp_arg2'].head())

def freqdist_wrapper(text):
    return FreqDist(text)

df['djp_arg3'] = df['djp_arg2'].apply(freqdist_wrapper)
print(df['djp_arg3'].head().apply(lambda x: x.most_common()))