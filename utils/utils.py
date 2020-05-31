from builtins import print
import numpy as np
import pandas as pd
import matplotlib
import random
import re

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import os
import operator
import utils

from utils.constants import PROPERTY_NAMES
from utils.constants import PREPARED_DATA_ROOT_DIRECTORY
from utils.constants import RESULTS_ROOT_DIRECTORY
from utils.constants import DATA_WEEKS_ROOT_DIRECORY
from utils.constants import DATA_PROPERTIES_ROOT_DIRECTORY
from utils.constants import TRAINING_TEST_DATA_RATIO
from utils.constants import ROOT_DIRECTORY

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from numpy.lib.npyio import loadtxt, savetxt
from numpy import genfromtxt
import csv
import shutil
from sklearn import preprocessing

def check_if_file_exits(file_name):
    return os.path.exists(file_name)

#Y: Slices Array taking all rows (:) but keeping only the first column (0)
#X: Slices Array taking all rows (:) but keeping all columns except the first (1:)
def readucr(filename, delimiter=','):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

#Y: Slices Array taking all rows (:) but keeping only the last column (-1)
#X: Slices Array taking all rows (:) but keeping all columns except the last one (:-1)
def readsits(filename, delimiter=','):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, -1]
    X = data[:, :-1]
    return X, Y

#Reads a csv file into a list
def read_csv_to_list(path, delimiter = ' '):
    dataset = pd.read_csv(path, sep=delimiter, header= None, engine='python')
    dataset = pd.DataFrame(dataset)
    dataset = dataset.values.tolist()

    return dataset

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None

        return directory_path

def clear_directory(directory_path):
    if not os.path.exists(directory_path):
        return
    else:
        shutil.rmtree(directory_path)





# Reads the prepared data for each referenced property into a dictionary
# e.g. datasets_dict['single'][0] returns the training data for the property single
def read_all_properties(root_dir):
    properties_dict = {}

    dataset_names_to_sort = []

    
    for property_name in PROPERTY_NAMES:
        root_dir_dataset = root_dir + PREPARED_DATA_ROOT_DIRECTORY + property_name + '/'
        file_name = root_dir_dataset + property_name
        x_train, y_train = readucr(file_name + '_train')
        x_test, y_test = readucr(file_name + '_test')

        properties_dict[property_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

        # dataset_names_to_sort.append((property_name, len(x_train)))

        # dataset_names_to_sort.sort(key=operator.itemgetter(1))

        # print(dataset_names_to_sort)

        # for i in range(len(PROPERTY_NAMES)):
        #     PROPERTY_NAMES[i] = dataset_names_to_sort[i][0]

    return properties_dict


def calculate_metrics(y_true, y_pred, duration):
    res = pd.DataFrame(data=np.zeros((1, 5), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'mcc', 'auc', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    #res['auc'] = roc_auc_score(y_true, y_pred, average='macro', multi_class="ovr")
    res['auc'] = roc_auc_score(preprocessing.binarize(np.array(y_true).reshape(-1,1)), y_pred, multi_class="ovr")
    res['mcc'] = matthews_corrcoef(y_true, y_pred)
    res['duration'] = duration
    print(res)
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)

#Transforms class labels into numerical values from 0 to n-1 classes
def transform_labels(y_train, y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # no validation split
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    y_train_test = np.concatenate((y_train, y_test), axis=0)
    # fit the encoder
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test

#generates a .csv-file with results for all properties combined
def generate_results_csv(output_file_name, root_dir, clfs):
    res = pd.DataFrame(data=np.zeros((0, 8), dtype=np.float), index=[],
                       columns=['property_name', 'iteration',
                                'precision', 'accuracy', 'recall', 'mcc','auc', 'duration'])

    properties_dict = read_all_properties(ROOT_DIRECTORY)
    for classifier_name in clfs:
        durr = 0.0

        for property_name in properties_dict.keys():
            output_dir = root_dir + '/results/' + property_name + '/' + 'df_metrics.csv'
            print(output_dir)
            if not os.path.exists(output_dir):
                continue
            df_metrics = pd.read_csv(output_dir)
            df_metrics['property_name'] = property_name
            df_metrics['iteration'] = 0
            res = pd.concat((res, df_metrics), axis=0, sort=False)
            durr += df_metrics['duration'][0]

    res.to_csv(root_dir + output_file_name, index=False)

    res = res.loc[res['classifier_name'].isin(clfs)]

    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs(output_directory, hist, y_pred, y_true, duration,
              lr=True, plot_test_acc=False):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    if plot_test_acc:
        df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    #df_best_model['best_model_train_acc'] = row_best_model['acc']
    if plot_test_acc:
        df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    if plot_test_acc:
        # plot losses
        plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


   
  
