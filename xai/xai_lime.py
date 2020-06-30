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
import heapq
import copy
np.random.seed(1)


class Lime:
    def __init__(self, property_name, class_names, num_features):
        self.property_name = str(property_name)
        self.class_names = class_names
        self.num_features = num_features
        
    def get_model(self):
        model = keras.models.load_model('./results/inception/' + self.property_name + '/best_model.hdf5', compile=False)

        return model
    
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
        datasets_dict = read_all_properties(root_dir, True)
        x_train = datasets_dict[self.property_name][0]
    
        if len(x_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

        return x_train
    
    def get_explainees(self):
        root_dir = ROOT_DIRECTORY
        datasets_dict = read_all_properties(root_dir, True)
        x_test = datasets_dict[self.property_name][2]

        if len(x_test.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        return x_test

    def get_label(self, household, model):
        prediction = model.predict(np.array([household]))
        if prediction[0,0] > prediction[0,1]:
            return 0
        else:
            return 1

    def create_explanations(self, save_plots, use_feature_names):
        model = self.get_model()
        train_data = self.get_training_data()[:300]
        test_data = self.get_explainees()
        feature_names = self.get_feature_names(train_data[0])
        if use_feature_names:
            explainer = lime.lime_tabular.RecurrentTabularExplainer(train_data, class_names=self.class_names, feature_names=feature_names, mode='classification')
        else:
            explainer = lime.lime_tabular.RecurrentTabularExplainer(train_data, class_names=self.class_names, mode='classification')
        counter = 0
        create_directory(ROOT_DIRECTORY +  'explanations_lime')
        if save_plots:
            for household in test_data:
                label = self.get_label(household, model)
                if counter < 10:
                    exp = explainer.explain_instance(np.array([household]), model.predict, num_features=self.num_features, top_labels=1, labels=label)
                    exp.save_to_file("explanations_lime/explanation_" + self.property_name + '_' +  str(counter) + ".html")
                counter = counter + 1
                if counter == 5:
                    break
        else:
            explanations = []
            print("Creating explanations for " + self.property_name + "(Lime")
            for household in test_data:
                label = self.get_label(household, model)
                if counter < 50:
                    exp = explainer.explain_instance(np.array([household]), model.predict, num_features=self.num_features, top_labels=1, labels=label)
                    explanations.append(exp.as_map())
                    print("explanation" + str(counter) + " done")
                counter = counter + 1
                if counter == 50:
                    return explanations
            
            
    def create_comparation_metrics(self):
        model = self.get_model()
        test_data = self.get_explainees()
        explanations = self.create_explanations(False, False)
        self.get_perturbation_analysis_data(explanations, test_data, model)
        self.get_stability_analysis_data(explanations, test_data, model)

       
    def get_perturbation_analysis_data(self, explanations, test_data, model):
        print("Starting Perturbation Analysis (LIME)")
        test_data_copy = copy.deepcopy(test_data)
        data_for_analysis = []
        for index, explanation in enumerate(explanations):
            if 1 in explanation:
                initial_classification = model.predict(np.array([test_data[index]])).flatten()[1]
                for item in explanation[1]:
                    if(item[1] < 0):
                        test_data_copy[index][item[0]] = 0
                post_perturbation_classification = model.predict(np.array([test_data_copy[index]])).flatten()[1]
                data_for_analysis.append([1, initial_classification, post_perturbation_classification, post_perturbation_classification - initial_classification ])
            else:
                continue
                # for item in explanation[0]:
                #     if(item[1] > 0):
                #         test_data[index][item[0]] = 0
                # print(model.predict(np.array([test_data[index]])))
            #print(test_data[index])
        data_for_analysis = pd.DataFrame(data_for_analysis)
        data_for_analysis.to_csv(ROOT_DIRECTORY + 'results/perturbation/' + self.property_name + '_lime.csv', index=False, header=False)
        print("Perturbation Analysis Done (LIME)")
        print(data_for_analysis)    

    def get_stability_analysis_data(self, explanations, test_data, model):
        print("Starting Stability Analysis (LIME)")
        data_for_analysis = [[],[]]
        for index, explanation in enumerate(explanations):
            if 1 in explanation:
                #print(explanation['1'])
                for index in range(0,3):
                    data_for_analysis[0].append(explanation[1][index][0])
                    data_for_analysis[1].append(test_data[index][explanation[1][index][0]][0])
                # for timeseries_index,item in enumerate(explanation['1']):
                #     if item > 0.02:
                #         print(timeseries_index)
                # post_perturbation_classification = model.predict(np.array([test_data[index]])).flatten()[1]
            else:
                continue
                # initial_classification = model.predict(np.array([test_data[index]])).flatten()[0]
                # for timeseries_index,item in enumerate(explanation['0']):
                #     if item < -0.01:
                #         test_data[index][timeseries_index] = 0
                # post_perturbation_classification = model.predict(np.array([test_data[index]])).flatten()[0]
                # data_for_analysis.append([0, initial_classification, post_perturbation_classification, post_perturbation_classification - initial_classification  ])
        data_for_analysis = pd.DataFrame(data_for_analysis)
        data_for_analysis.to_csv(ROOT_DIRECTORY + 'results/stability/' + self.property_name + '_lime.csv', index=False, header=False)
        print("Stability Analysis Done (LIME)")
        # data_for_analysis = pd.DataFrame(data_for_analysis)
        # data_for_analysis.to_csv(ROOT_DIRECTORY + 'results/perturbation/' + self.property_name + '_shap.csv') 











