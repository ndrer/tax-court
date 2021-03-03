import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB

#Prep
df = pd.read_csv(r'D:\Blag - DATA\tax_court_stem_2.csv',header=0, converters={'no_putusan':str,'jenis_pajak':str,'sengketa':str,'djp_arg':str,'wp_arg':str,'pdpt_majelis':str})
df = df.apply(lambda x: x.astype(str).str.lower())

features_djp = df['djp_arg'].values
features_wp = df['wp_arg'].values
features_maj = df['argumen'].values

labels = df.iloc[:, 7].values

vectorizer = TfidfVectorizer (lowercase=False, max_features=2500, min_df=7, max_df=0.8)
processed_features_djp = vectorizer.fit_transform(features_djp).toarray()
processed_features_wp = vectorizer.fit_transform(features_wp).toarray()
processed_features_maj = vectorizer.fit_transform(features_maj).toarray()



#reguler random forest, SVM, and Naive Bayes
X_train1, X_test1, y_train1, y_test1 = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X_train1, y_train1)
predictions = rf.predict(X_test1)
print(confusion_matrix(y_test1,predictions))
print(classification_report(y_test1,predictions))
print(accuracy_score(y_test1, predictions))

X_train2, X_test2, y_train2, y_test2 = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train2, y_train2)
y_pred2 = svclassifier.predict(X_test2)
print(f1_score(y_test2, y_pred2, average='weighted'))

X_train3, X_test3, y_train3, y_test3 = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
cnb = ComplementNB()
cnb.fit(X_train3, y_train3)
y_pred5 = cnb.predict(X_test3)
print(f1_score(y_test3, y_pred5, average='weighted'))

#DJP Argument
X_train4, X_test4, y_train4, y_test4 = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X_train4, y_train4)
predictions = rf.predict(X_test4)
print(confusion_matrix(y_test4,predictions))
print(classification_report(y_test4,predictions))
print(accuracy_score(y_test4, predictions))

X_train5, X_test5, y_train5, y_test5 = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train5, y_train5)
y_pred2 = svclassifier.predict(X_test5)
print(f1_score(y_test5, y_pred2, average='weighted'))

X_train6, X_test6, y_train6, y_test6 = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
cnb = ComplementNB()
cnb.fit(X_train6, y_train6)
y_pred5 = cnb.predict(X_test6)
print(f1_score(y_test6, y_pred5, average='weighted'))

#WP Argument
X_train7, X_test7, y_train7, y_test7 = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X_train7, y_train7)
predictions = rf.predict(X_test7)
print(confusion_matrix(y_test7,predictions))
print(classification_report(y_test7,predictions))
print(accuracy_score(y_test7, predictions))

X_train8, X_test8, y_train8, y_test8 = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train8, y_train8)
y_pred2 = svclassifier.predict(X_test8)
print(f1_score(y_test8, y_pred2, average='weighted'))

X_train9, X_test9, y_train9, y_test9 = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
cnb = ComplementNB()
cnb.fit(X_train9, y_train9)
y_pred5 = cnb.predict(X_test9)
print(f1_score(y_test9, y_pred5, average='weighted'))