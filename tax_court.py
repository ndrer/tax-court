import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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
rf = RandomForestClassifier(n_estimators=200, random_state=0,class_weight='balanced')
rf.fit(X_train1, y_train1)
y_pred1 = rf.predict(X_test1)
print(confusion_matrix(y_test1,y_pred1))
print(classification_report(y_test1,y_pred1))
print(accuracy_score(y_test1, y_pred1))

X_train2, X_test2, y_train2, y_test2 = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
svclassifier = SVC(kernel='linear',class_weight='balanced')
svclassifier.fit(X_train2, y_train2)
y_pred2 = svclassifier.predict(X_test2)
print(confusion_matrix(y_test2,y_pred2))
print(classification_report(y_test2,y_pred2))
print(accuracy_score(y_test2, y_pred2))

X_train3, X_test3, y_train3, y_test3 = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
cnb = ComplementNB()
cnb.fit(X_train3, y_train3)
y_pred3 = cnb.predict(X_test3)
print(confusion_matrix(y_test3,y_pred3))
print(classification_report(y_test3,y_pred3))
print(accuracy_score(y_test3, y_pred3))

#DJP Argument
X_train4, X_test4, y_train4, y_test4 = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
rf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight='balanced')
rf.fit(X_train4, y_train4)
y_pred4 = rf.predict(X_test4)
print(confusion_matrix(y_test4,y_pred4))
print(classification_report(y_test4,y_pred4))
print(accuracy_score(y_test4, y_pred4))

X_train5, X_test5, y_train5, y_test5 = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
svclassifier = SVC(kernel='linear', class_weight='balanced')
svclassifier.fit(X_train5, y_train5)
y_pred5 = svclassifier.predict(X_test5)
print(confusion_matrix(y_test5,y_pred5))
print(classification_report(y_test5,y_pred5))
print(accuracy_score(y_test5, y_pred5))


X_train6, X_test6, y_train6, y_test6 = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
cnb = ComplementNB()
cnb.fit(X_train6, y_train6)
y_pred6 = cnb.predict(X_test6)
print(confusion_matrix(y_test6,y_pred6))
print(classification_report(y_test6,y_pred6))
print(accuracy_score(y_test6, y_pred6))

#WP Argument
X_train7, X_test7, y_train7, y_test7 = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
rf = RandomForestClassifier(n_estimators=200, random_state=0,class_weight='balanced')
rf.fit(X_train7, y_train7)
y_pred7 = rf.predict(X_test7)
print(confusion_matrix(y_test7,y_pred7))
print(classification_report(y_test7,y_pred7))
print(accuracy_score(y_test7, y_pred7))

X_train8, X_test8, y_train8, y_test8 = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
svclassifier = SVC(kernel='linear',class_weight='balanced')
svclassifier.fit(X_train8, y_train8)
y_pred8 = svclassifier.predict(X_test8)
print(confusion_matrix(y_test8,y_pred8))
print(classification_report(y_test8,y_pred8))
print(accuracy_score(y_test8, y_pred8))

X_train9, X_test9, y_train9, y_test9 = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
cnb = ComplementNB()
cnb.fit(X_train9, y_train9)
y_pred9 = cnb.predict(X_test9)
print(confusion_matrix(y_test9,y_pred9))
print(classification_report(y_test9,y_pred9))
print(accuracy_score(y_test9, y_pred9))

#trying Logistic
X_train10, X_test10, y_train10, y_test10 = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
logis = LogisticRegression(random_state=0, solver='lbfgs',multi_class='auto',class_weight='balanced')
logis.fit(X_train10, y_train10)
y_pred10 = logis.predict(X_test10)
print(confusion_matrix(y_test10,y_pred10))
print(classification_report(y_test10,y_pred10))
print(accuracy_score(y_test10, y_pred10))

#trying kNN
X_train11, X_test11, y_train11, y_test11 = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train11, y_train11)
y_pred11 = neigh.predict(X_test11)
print(confusion_matrix(y_test11,y_pred11))
print(classification_report(y_test11,y_pred11))
print(accuracy_score(y_test11, y_pred11))