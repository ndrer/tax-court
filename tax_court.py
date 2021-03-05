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


#reguler logistic, kNN, random forest, SVM, and Naive Bayes
X_train_log_all, X_test_log_all, y_train_log_all, y_test_log_all = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
logis = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',class_weight='balanced')
logis.fit(X_train_log_all, y_train_log_all)
y_pred_log_all = logis.predict(X_test_log_all)
print(confusion_matrix(y_test_log_all,y_pred_log_all))
print(classification_report(y_test_log_all,y_pred_log_all))
print(accuracy_score(y_test_log_all, y_pred_log_all))

X_train_knn_all, X_test_knn_all, y_train_knn_all, y_test_knn_all = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train_knn_all, y_train_knn_all)
y_pred_knn_all = neigh.predict(X_test_knn_all)
print(confusion_matrix(y_test_knn_all,y_pred_knn_all))
print(classification_report(y_test_knn_all,y_pred_knn_all))
print(accuracy_score(y_test_knn_all, y_pred_knn_all))

X_train_rf_all, X_test_rf_all, y_train_rf_all, y_test_rf_all = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
rf = RandomForestClassifier(n_estimators=200, random_state=0,class_weight='balanced')
rf.fit(X_train_rf_all, y_train_rf_all)
y_pred_rf_all = rf.predict(X_test_rf_all)
print(confusion_matrix(y_test_rf_all,y_pred_rf_all))
print(classification_report(y_test_rf_all,y_pred_rf_all))
print(accuracy_score(y_test_rf_all, y_pred_rf_all))

X_train_svm_all, X_test_svm_all, y_train_svm_all, y_test_svm_all = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
svclassifier = SVC(kernel='linear',class_weight='balanced')
svclassifier.fit(X_train_svm_all, y_train_svm_all)
y_pred_svm_all = svclassifier.predict(X_test_svm_all)
print(confusion_matrix(y_test_svm_all,y_pred_svm_all))
print(classification_report(y_test_svm_all,y_pred_svm_all))
print(accuracy_score(y_test_svm_all, y_pred_svm_all))

X_train_nb_all, X_test_nb_all, y_train_nb_all, y_test_nb_all = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
cnb = ComplementNB()
cnb.fit(X_train_nb_all, y_train_nb_all)
y_pred_nb_all = cnb.predict(X_test_nb_all)
print(confusion_matrix(y_test_nb_all,y_pred_nb_all))
print(classification_report(y_test_nb_all,y_pred_nb_all))
print(accuracy_score(y_test_nb_all, y_pred_nb_all))

#DJP Argument
#DJP Argument
X_train_log_djp, X_test_log_djp, y_train_log_djp, y_test_log_djp = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
logis = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',class_weight='balanced')
logis.fit(X_train_log_djp, y_train_log_djp)
y_pred_log_djp = logis.predict(X_test_log_djp)
print(confusion_matrix(y_test_log_djp,y_pred_log_djp))
print(classification_report(y_test_log_djp,y_pred_log_djp))
print(accuracy_score(y_test_log_djp, y_pred_log_djp))

X_train_knn_djp, X_test_knn_djp, y_train_knn_djp, y_test_knn_djp = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train_knn_djp, y_train_knn_djp)
y_pred_knn_djp = neigh.predict(X_test_knn_djp)
print(confusion_matrix(y_test_knn_djp,y_pred_knn_djp))
print(classification_report(y_test_knn_djp,y_pred_knn_djp))
print(accuracy_score(y_test_knn_djp, y_pred_knn_djp))

X_train_rf_djp, X_test_rf_djp, y_train_rf_djp, y_test_rf_djp = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
rf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight='balanced')
rf.fit(X_train_rf_djp, y_train_rf_djp)
y_pred_rf_djp = rf.predict(X_test_rf_djp)
print(confusion_matrix(y_test_rf_djp,y_pred_rf_djp))
print(classification_report(y_test_rf_djp,y_pred_rf_djp))
print(accuracy_score(y_test_rf_djp, y_pred_rf_djp))

X_train_svm_djp, X_test_svm_djp, y_train_svm_djp, y_test_svm_djp = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
svclassifier = SVC(kernel='linear', class_weight='balanced')
svclassifier.fit(X_train_svm_djp, y_train_svm_djp)
y_pred_svm_djp = svclassifier.predict(X_test_svm_djp)
print(confusion_matrix(y_test_svm_djp,y_pred_svm_djp))
print(classification_report(y_test_svm_djp,y_pred_svm_djp))
print(accuracy_score(y_test_svm_djp, y_pred_svm_djp))

X_train_nb_djp, X_test_nb_djp, y_train_nb_djp, y_test_nb_djp = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
cnb = ComplementNB()
cnb.fit(X_train_nb_djp, y_train_nb_djp)
y_pred_nb_djp = cnb.predict(X_test_nb_djp)
print(confusion_matrix(y_test_nb_djp,y_pred_nb_djp))
print(classification_report(y_test_nb_djp,y_pred_nb_djp))
print(accuracy_score(y_test_nb_djp, y_pred_nb_djp))

#WP Argument
X_train_log_wp, X_test_log_wp, y_train_log_wp, y_test_log_wp = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
logis = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',class_weight='balanced')
logis.fit(X_train_log_wp, y_train_log_wp)
y_pred_log_wp = logis.predict(X_test_log_wp)
print(confusion_matrix(y_test_log_wp,y_pred_log_wp))
print(classification_report(y_test_log_wp,y_pred_log_wp))
print(accuracy_score(y_test_log_wp, y_pred_log_wp))

X_train_knn_wp, X_test_knn_wp, y_train_knn_wp, y_test_knn_wp = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train_knn_wp, y_train_knn_wp)
y_pred_knn_wp = neigh.predict(X_test_knn_wp)
print(confusion_matrix(y_test_knn_wp,y_pred_knn_wp))
print(classification_report(y_test_knn_wp,y_pred_knn_wp))
print(accuracy_score(y_test_knn_wp, y_pred_knn_wp))

X_train_rf_wp, X_test_rf_wp, y_train_rf_wp, y_test_rf_wp = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
rf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight='balanced')
rf.fit(X_train_rf_wp, y_train_rf_wp)
y_pred_rf_wp = rf.predict(X_test_rf_wp)
print(confusion_matrix(y_test_rf_wp,y_pred_rf_wp))
print(classification_report(y_test_rf_wp,y_pred_rf_wp))
print(accuracy_score(y_test_rf_wp, y_pred_rf_wp))

X_train_svm_wp, X_test_svm_wp, y_train_svm_wp, y_test_svm_wp = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
svclassifier = SVC(kernel='linear', class_weight='balanced')
svclassifier.fit(X_train_svm_wp, y_train_svm_wp)
y_pred_svm_wp = svclassifier.predict(X_test_svm_wp)
print(confusion_matrix(y_test_svm_wp,y_pred_svm_wp))
print(classification_report(y_test_svm_wp,y_pred_svm_wp))
print(accuracy_score(y_test_svm_wp, y_pred_svm_wp))

X_train_nb_wp, X_test_nb_wp, y_train_nb_wp, y_test_nb_wp = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
cnb = ComplementNB()
cnb.fit(X_train_nb_wp, y_train_nb_wp)
y_pred_nb_wp = cnb.predict(X_test_nb_wp)
print(confusion_matrix(y_test_nb_wp,y_pred_nb_wp))
print(classification_report(y_test_nb_wp,y_pred_nb_wp))
print(accuracy_score(y_test_nb_wp, y_pred_nb_wp))