import numpy as np
import scipy, matplotlib.pyplot as plt, IPython.display as ipd
import sklearn, pandas as pd
import librosa, librosa.display
import sys
import time
from pathlib import Path
import urllib
import librosa.feature as lf
from sklearn.metrics import classification_report, confusion_matrix
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from functions import *
import warnings
warnings.filterwarnings('ignore')

every_composer = ['albeniz', 'beethoven', 'brahms', 'chopin', 'clementi', 'debussy', 'grieg', 'haydn', 'liszt','mendel', 'mozart', 'rachmaninoff', 'schubert', 'schumman','tchaikovsky']
def folder_to_features(folder_name):
    fcl = [lf.spectral_centroid,
    lf.spectral_bandwidth,
    lf.spectral_contrast,
    lf.spectral_rolloff,
    lf.mfcc,
    lf.zero_crossing_rate]
    
    n_mfcc = 12
    audio_time_series_list = []
    song_count = 0
    
    mp3_names =[
        str(p)[len(f'{folder_name}/'):-len('.mp3')] for p in Path().glob(f'{folder_name}/*.mp3')
    ]
    
    for p in Path().glob(f'{folder_name}/*.mp3'):  
        for i in range(10):
            x = librosa.load(p, duration = 5, offset = 5+i*20)[0]
            if x.shape != (0,):
                audio_time_series_list.append(x)
    for song in audio_time_series_list: 
        for f in fcl:
            if f == lf.spectral_centroid:
                feature_i = f(y=song).T
                feature_i = np.hstack((np.mean(feature_i, axis = 0), np.std(feature_i, axis = 0)))
            else:
                current = f(y=song).T
                feature_i = np.hstack((feature_i, np.mean(current, axis = 0), np.std(current, axis = 0)))
        if song_count == 0:
            total_array = feature_i
        else:
            total_array = np.vstack((total_array, feature_i))
        song_count += 1
    label_array = np.full((total_array.shape[0],1), folder_name)
    total_array = np.hstack((label_array, total_array))
    return total_array

def combine_multiple_folder_sets(folder_name_list):
    for folder_data in folder_name_list:
        if folder_data == folder_name_list[0]:
            all_data = folder_to_features(folder_data)
        else:
            all_data = np.vstack((all_data, folder_to_features(folder_data)))
    return all_data

def separate_and_scale_features_and_labels(all_data, frac = 0.05):
    labels, features = np.split(all_data,[1], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, labels,stratify=labels)
    scaler = sklearn.preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

loss_functions = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron','squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
def try_sgdc_with_lf(X_train_scaled, X_test_scaled, y_train, y_test, lf = 'hinge'):
    model = SGDClassifier(loss=lf)
    model.fit(X_train_scaled, y_train)
    return model, model.score(X_test_scaled, y_test)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
