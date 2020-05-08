from __future__ import print_function
import keras
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
np.random.seed(1)

class Lime:
    def __init__(self, training_data, explainee, prediction_function):
        self.training_data = training_data
        self.explainee = explainee
        self.prediction_function = prediction_function
    
    def get_model(self):
        return keras.models.load_model('./results/inception/_itr_1/single/last_model.hdf5')

    def get_explanation(self):
        model = self.get_model()
        print(model)
        # explainer = lime.lime_tabular.LimeTabularExplainer(self.training_data, feature_names=range(len(self.training_data)), discretize_continuous=True)
        # exp = explainer.explain_instance(explainee, prediction_function, num_features=2, top_labels=1)
        
        return 'exp'

Lime('test', 'test', 'test')
