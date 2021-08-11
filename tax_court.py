import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tabulate import tabulate

#Prep
df = pd.read_csv(r'D:\Blag - DATA\tax_court_stem_3.csv',header=0, converters={'no_putusan':str,'jenis_pajak':str,'sengketa':str,'djp_arg':str,'wp_arg':str,'pdpt_majelis':str})
df = df.apply(lambda x: x.astype(str).str.lower())

features_djp = df['djp_arg'].values
features_wp = df['wp_arg'].values
features_both = df['argumen'].values

labels = df.iloc[:, 7].values

vectorizer = TfidfVectorizer (lowercase=False, max_features=2500, min_df=7, max_df=0.8)
processed_features_djp = vectorizer.fit_transform(features_djp).toarray()
processed_features_wp = vectorizer.fit_transform(features_wp).toarray()
processed_features_both = vectorizer.fit_transform(features_both).toarray()

#Classifier
logis = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',class_weight='balanced')
neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
rf = RandomForestClassifier(n_estimators=50, random_state=0, class_weight='balanced')
svclassifier = SVC(kernel='linear', class_weight='balanced')
cnb = ComplementNB(alpha=1.0e-10)
xgb = XGBClassifier(objective = 'multi:softmax', num_class = 3, eta=0.1)

#DJP Argument
X_train_djp, X_test_djp, y_train_djp, y_test_djp = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)

logis.fit(X_train_djp, y_train_djp)
y_pred_log_djp = logis.predict(X_test_djp)
print(confusion_matrix(y_test_djp,y_pred_log_djp))
print(classification_report(y_test_djp,y_pred_log_djp))

neigh.fit(X_train_djp, y_train_djp)
y_pred_knn_djp = neigh.predict(X_test_djp)
print(confusion_matrix(y_test_djp,y_pred_knn_djp))
print(classification_report(y_test_djp,y_pred_knn_djp))

rf.fit(X_train_djp, y_train_djp)
y_pred_rf_djp = rf.predict(X_test_djp)
print(confusion_matrix(y_test_djp,y_pred_rf_djp))
print(classification_report(y_test_djp,y_pred_rf_djp))

svclassifier.fit(X_train_djp, y_train_djp)
y_pred_svm_djp = svclassifier.predict(X_test_djp)
print(confusion_matrix(y_test_djp,y_pred_svm_djp))
print(classification_report(y_test_djp,y_pred_svm_djp))

cnb.fit(X_train_djp, y_train_djp)
y_pred_nb_djp = cnb.predict(X_test_djp)
print(confusion_matrix(y_test_djp,y_pred_nb_djp))
print(classification_report(y_test_djp,y_pred_nb_djp))

xgb.fit(X_train_djp, y_train_djp)
y_pred_xgb_djp = xgb.predict(X_test_djp)
print(confusion_matrix(y_test_djp,y_pred_xgb_djp))
print(classification_report(y_test_djp,y_pred_xgb_djp))


#Taxpayer argument

X_train_wp, X_test_wp, y_train_wp, y_test_wp = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)

logis.fit(X_train_wp, y_train_wp)
y_pred_log_wp = logis.predict(X_test_wp)
print(confusion_matrix(y_test_wp,y_pred_log_wp))
print(classification_report(y_test_wp,y_pred_log_wp))

neigh.fit(X_train_wp, y_train_wp)
y_pred_knn_wp = neigh.predict(X_test_wp)
print(confusion_matrix(y_test_wp,y_pred_knn_wp))
print(classification_report(y_test_wp,y_pred_knn_wp))

rf.fit(X_train_wp, y_train_wp)
y_pred_rf_wp = rf.predict(X_test_wp)
print(confusion_matrix(y_test_wp,y_pred_rf_wp))
print(classification_report(y_test_wp,y_pred_rf_wp))

svclassifier.fit(X_train_wp, y_train_wp)
y_pred_svm_wp = svclassifier.predict(X_test_wp)
print(confusion_matrix(y_test_wp,y_pred_svm_wp))
print(classification_report(y_test_wp,y_pred_svm_wp))

cnb.fit(X_train_wp, y_train_wp)
y_pred_nb_wp = cnb.predict(X_test_wp)
print(confusion_matrix(y_test_wp,y_pred_nb_wp))
print(classification_report(y_test_wp,y_pred_nb_wp))

xgb.fit(X_train_wp, y_train_wp)
y_pred_xgb_wp = xgb.predict(X_test_wp)
print(confusion_matrix(y_test_wp,y_pred_xgb_wp))
print(classification_report(y_test_wp,y_pred_xgb_wp))

#Both arguments

X_train_both, X_test_both, y_train_both, y_test_both = train_test_split(processed_features_both, labels, test_size=0.2, random_state=1, stratify=labels)

logis.fit(X_train_both, y_train_both)
y_pred_log_both = logis.predict(X_test_both)
print(confusion_matrix(y_test_both,y_pred_log_both))
print(classification_report(y_test_both,y_pred_log_both))

neigh.fit(X_train_both, y_train_both)
y_pred_knn_both = neigh.predict(X_test_both)
print(confusion_matrix(y_test_both,y_pred_knn_both))
print(classification_report(y_test_both,y_pred_knn_both))

rf.fit(X_train_both, y_train_both)
y_pred_rf_both = rf.predict(X_test_both)
print(confusion_matrix(y_test_both,y_pred_rf_both))
print(classification_report(y_test_both,y_pred_rf_both))

svclassifier.fit(X_train_both, y_train_both)
y_pred_svm_both = svclassifier.predict(X_test_both)
print(confusion_matrix(y_test_both,y_pred_svm_both))
print(classification_report(y_test_both,y_pred_svm_both))

cnb.fit(X_train_both, y_train_both)
y_pred_nb_both = cnb.predict(X_test_both)
print(confusion_matrix(y_test_both,y_pred_nb_both))
print(classification_report(y_test_both,y_pred_nb_both))

xgb.fit(X_train_both, y_train_both)
y_pred_xgb_both = xgb.predict(X_test_both)
print(confusion_matrix(y_test_both,y_pred_xgb_both))
print(classification_report(y_test_both,y_pred_xgb_both))

#tabulating F1 Score
tabout_header = ["Classifier","DJP Argument","WP Argument", "Both"]
tabout =[
        ["Logistic regression",round(f1_score(y_test_djp, y_pred_log_djp, average='weighted'),3),
                 round(f1_score(y_test_wp, y_pred_log_wp, average='weighted'),3),
                 round(f1_score(y_test_both, y_pred_log_both, average='weighted'),3)
                 ],
        ["kNN",round(f1_score(y_test_djp, y_pred_knn_djp, average='weighted'),3),
                 round(f1_score(y_test_wp, y_pred_knn_wp, average='weighted'),3),
                 round(f1_score(y_test_both, y_pred_knn_both, average='weighted'),3)
                 ],
        ["Random forest",round(f1_score(y_test_djp, y_pred_rf_djp, average='weighted'),3),
                 round(f1_score(y_test_wp, y_pred_rf_wp, average='weighted'),3),
                 round(f1_score(y_test_both, y_pred_rf_both, average='weighted'),3)
                 ],
        ["SVM",round(f1_score(y_test_djp, y_pred_svm_djp, average='weighted'),3),
                 round(f1_score(y_test_wp, y_pred_svm_wp, average='weighted'),3),
                 round(f1_score(y_test_both, y_pred_svm_both, average='weighted'),3)
                 ],
        ["Naive Bayes",round(f1_score(y_test_djp, y_pred_nb_djp, average='weighted'),3),
                 round(f1_score(y_test_wp, y_pred_nb_wp, average='weighted'),3),
                 round(f1_score(y_test_both, y_pred_nb_both, average='weighted'),3)
                 ],
        ["XGBoost",round(f1_score(y_test_djp, y_pred_xgb_djp, average='weighted'),3),
                 round(f1_score(y_test_wp, y_pred_xgb_wp, average='weighted'),3),
                 round(f1_score(y_test_both, y_pred_xgb_both, average='weighted'),3)
                 ]
        ]
print(tabulate(tabout,tabout_header, tablefmt='pretty'))
print(tabulate(tabout,tabout_header, tablefmt='html'))

