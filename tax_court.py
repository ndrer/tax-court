import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#Prep
df = pd.read_csv(r'D:\Blag - DATA\tax_court_stem_2.csv',header=0, converters={'no_putusan':str,'jenis_pajak':str,'sengketa':str,'djp_arg':str,'wp_arg':str,'pdpt_majelis':str})
df = df.apply(lambda x: x.astype(str).str.lower())


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