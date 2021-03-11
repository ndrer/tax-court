# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 17:11:54 2021

@author: Andreas
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras.regularizers import l1
from tensorflow.keras import backend as K
K.clear_session()

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

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

#all argument
X_train_ann_all, X_test_ann_all, y_train_ann_all, y_test_ann_all = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
input_dim = X_train_ann_all.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu',activity_regularizer=l1(0.0001)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history_all = model.fit(X_train_ann_all, y_train_ann_all, epochs=50,verbose=False,validation_data=(X_test_ann_all, y_test_ann_all), batch_size=12)


loss, accuracy = model.evaluate(X_train_ann_all, y_train_ann_all, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_ann_all, y_test_ann_all, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history_all)

#DJP Argument
X_train_ann_djp, X_test_ann_djp, y_train_ann_djp, y_test_ann_djp = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
input_dim = X_train_ann_djp.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu',activity_regularizer=l1(0.0001)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history_djp = model.fit(X_train_ann_djp, y_train_ann_djp, epochs=50,verbose=False,validation_data=(X_test_ann_djp, y_test_ann_djp), batch_size=10)


loss, accuracy = model.evaluate(X_train_ann_djp, y_train_ann_djp, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_ann_djp, y_test_ann_djp, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history_djp)

#WP Argument
X_train_ann_wp, X_test_ann_wp, y_train_ann_wp, y_test_ann_wp = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
input_dim = X_train_ann_wp.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu',activity_regularizer=l1(0.0001)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history_wp = model.fit(X_train_ann_wp, y_train_ann_wp, epochs=50,verbose=False,validation_data=(X_test_ann_wp, y_test_ann_wp), batch_size=10)


loss, accuracy = model.evaluate(X_train_ann_wp, y_train_ann_wp, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_ann_wp, y_test_ann_wp, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history_wp)