import sklearn
import shap
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
import sys
sys.path.insert(1, '../')
from utils.utils import create_directory, read_all_properties
from utils.constants import ROOT_DIRECTORY
import copy
import numpy as np
import pandas as pd
import heapq
from xai.xai_plots import create_plot_shap


class Shap:
    def __init__(self, property_name, class_names, num_features):
        self.property_name = str(property_name)
        self.class_names = class_names
        self.num_features = num_features
        
    def get_model(self):
        model = keras.models.load_model('./results5/inception/' + self.property_name + '/best_model.hdf5', compile=False)

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
            #add a dimension to make it multivariate with one dimension
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        return x_test

    def get_label(self, household):
        prediction = self.get_model().predict(np.array([household]))
        if prediction[0,0] > prediction[0,1]:
            return 0
        else:
            return 1

    def create_explanations(self, save_plots, use_feature_names):
        train_data = self.get_training_data()
        test_data = self.get_explainees()
        model = self.get_model()
        feature_names = self.get_feature_names(train_data[0])
        sess = K.get_session()
        explainer = shap.DeepExplainer(model, train_data[:100], sess)
        counter = 0
        create_directory(ROOT_DIRECTORY +  'explanations_shap')
        if save_plots:
            for household in test_data:
                label = self.get_label(household)
                if counter < 10 and label == 1 and self.get_model().predict(np.array([household]))[0,1] > 0.9:
                    shap_values = explainer.shap_values(np.array([household]))
                    if counter == 4:
                        create_plot_shap(shap_values[1][0].flatten(), self.property_name, counter)
                    #shap.save_html('explanations_shap/explanation_shap_' + self.property_name + '_' + str(counter) +  '.html', shap.force_plot(explainer.expected_value[self.get_label(household)], shap_values[self.get_label(household)][0].flatten(), show=False, features=household.flatten(), feature_names=feature_names))
                counter = counter + 1
                if counter == 5:
                    break
        else:
            explanations = []
            print("Creating explanations for " + self.property_name + "(Shap")
            for household in test_data:
                label = self.get_label(household)
                if counter < 50:
                    shap_values = explainer.shap_values(np.array([household]))
                    explanations.append({str(label): shap_values[self.get_label(household)][0].flatten()})
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
        print("Starting Perturbation Analysis (SHAP)")
        test_data_copy = copy.deepcopy(test_data)
        data_for_analysis = []      
        for index, explanation in enumerate(explanations):
            if '1' in explanation:
                initial_classification = model.predict(np.array([test_data[index]])).flatten()[1]
                for timeseries_index,item in enumerate(explanation['1']):
                    if item > 0.02:
                        test_data_copy[index][timeseries_index] = 0
                post_perturbation_classification = model.predict(np.array([test_data_copy[index]])).flatten()[1]
                data_for_analysis.append([1, initial_classification, post_perturbation_classification, post_perturbation_classification - initial_classification ])
            else:
                continue
                # initial_classification = model.predict(np.array([test_data[index]])).flatten()[0]
                # for timeseries_index,item in enumerate(explanation['0']):
                #     if item < -0.01:
                #         test_data[index][timeseries_index] = 0
                # post_perturbation_classification = model.predict(np.array([test_data[index]])).flatten()[0]
                # data_for_analysis.append([0, initial_classification, post_perturbation_classification, post_perturbation_classification - initial_classification  ])
        data_for_analysis = pd.DataFrame(data_for_analysis)
        data_for_analysis.to_csv(ROOT_DIRECTORY + 'results/perturbation/' + self.property_name + '_shap.csv', index=False, header=False)
        print("Perturbation Analysis Done (SHAP)")
        
    
    def get_stability_analysis_data(self, explanations, test_data, model):
        print("Starting Stability Analysis (SHAP)")
        data_for_analysis = [[],[]]
        for index, explanation in enumerate(explanations):
            if '1' in explanation:
                #print(explanation['1'])
                for value in heapq.nlargest(3, explanation['1']):
                    data_for_analysis[0].append(np.where(explanation['1'] == value)[0][0])
                    print(test_data[index][np.where(explanation['1'] == value)[0][0]])
                    print(np.where(explanation['1'] == value))
                    print(test_data[index])
                    print(test_data[index][112])
                    data_for_analysis[1].append(test_data[index][np.where(explanation['1'] == value)[0][0]][0])
                    #print(explanation['1'].index(value))
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
        data_for_analysis.to_csv(ROOT_DIRECTORY + 'results/stability/' + self.property_name + '_shap.csv', index=False, header=False) 
        print("Stability Analysis Done (SHAP)")
        # data_for_analysis = pd.DataFrame(data_for_analysis)
        # data_for_analysis.to_csv(ROOT_DIRECTORY + 'results/perturbation/' + self.property_name + '_shap.csv') 


