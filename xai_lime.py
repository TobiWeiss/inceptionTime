from __future__ import print_function
import keras
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
import pandas as pd
from utils.constants import ROOT_DIRECTORY
from utils.utils import create_directory, read_all_properties, transform_labels
np.random.seed(1)


class Lime:
    def __init__(self, property_name, model, class_names, num_features):
        self.property_name = str(property_name)
        self.model = model
        self.class_names = class_names
        self.num_features = num_features
        
    def get_model(self):
        model = keras.models.load_model('./results/inception/_itr_1/single/last_model.hdf5', compile=False)
    
    def get_feature_names(self, explainee):
        feature_names = list()
        for index in range(len(explainee)):
            if index < 48:
                day = "Mo"
            elif index < 96:
                day = "Tue"
            elif index < 144:
                day = "Wed"
            elif index < 192:
                day = "Thur"
            elif index < 240:
                day = "Frid"
            elif index < 288:
                day = "Sat"
            else:
                day = "Sun"
            time = (((index / 48) % 1) * 24) + 0.5
            feature_names.append(str(day) + "_" + str(round(time,1)))
       
        return feature_names
    
    def get_training_data(self):
        root_dir = ROOT_DIRECTORY
        datasets_dict = read_all_properties(root_dir)
        x_train = datasets_dict[self.property_name][0]
    
        if len(x_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

        return x_train
    
    def get_explainees(self):
        root_dir = ROOT_DIRECTORY
        datasets_dict = read_all_properties(root_dir)
        x_test = datasets_dict[self.property_name][2]

        if len(x_test.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        return x_test

    def get_label(self, household):
        prediction = self.model.predict(np.array([household]))
        if prediction[0,0] > prediction[0,1]:
            return 0
        else:
            return 1

    def create_explanations(self):
        model = self.get_model()
        train_data = self.get_training_data()
        test_data = self.get_explainees()
        feature_names = self.get_feature_names(train_data[0])
        explainer = lime.lime_tabular.RecurrentTabularExplainer(train_data, class_names=self.class_names, feature_names=feature_names, mode='classification')
        counter = 0
        label = self.get_label(test_data[0])
        create_directory(ROOT_DIRECTORY +  'explanations')
        for household in test_data:
            if counter == 7:
                print(model.predict(np.array([household])))
            if counter < 10:
                exp = explainer.explain_instance(np.array([household]), self.model.predict, num_features=self.num_features, top_labels=1, labels=0)
                exp.save_to_file("explanations/explanation_" +  str(counter) + ".html")
            counter = counter + 1






