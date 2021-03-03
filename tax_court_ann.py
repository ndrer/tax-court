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

X_train10, X_test10, y_train10, y_test10 = train_test_split(processed_features_maj, labels, test_size=0.2, random_state=1, stratify=labels)
input_dim = X_train10.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu',activity_regularizer=l1(0.0001)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history1 = model.fit(X_train10, y_train10, epochs=100,verbose=False,validation_data=(X_test10, y_test10), batch_size=10)


loss, accuracy = model.evaluate(X_train10, y_train10, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test10, y_test10, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history1)

#DJP Argument
X_train11, X_test11, y_train11, y_test11 = train_test_split(processed_features_djp, labels, test_size=0.2, random_state=1, stratify=labels)
input_dim = X_train11.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu',activity_regularizer=l1(0.0001)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history2 = model.fit(X_train11, y_train11, epochs=100,verbose=False,validation_data=(X_test11, y_test11), batch_size=10)


loss, accuracy = model.evaluate(X_train11, y_train11, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test11, y_test11, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history2)

#WP Argument
X_train12, X_test12, y_train12, y_test12 = train_test_split(processed_features_wp, labels, test_size=0.2, random_state=1, stratify=labels)
input_dim = X_train12.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu',activity_regularizer=l1(0.0001)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history3 = model.fit(X_train12, y_train12, epochs=100,verbose=False,validation_data=(X_test12, y_test12), batch_size=10)


loss, accuracy = model.evaluate(X_train12, y_train12, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test12, y_test12, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history3)
