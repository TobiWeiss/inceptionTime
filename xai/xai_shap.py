import sklearn
import shap
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
import sys
sys.path.insert(1, '../')
from utils.utils import create_directory, read_all_properties
from utils.constants import ROOT_DIRECTORY
import numpy as np

class Shap:
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
            #add a dimension to make it multivariate with one dimension
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        return x_test

    def get_label(self, household):
        prediction = self.get_model().predict(np.array([household]))
        if prediction[0,0] > prediction[0,1]:
            return 0
        else:
            return 1

    def create_explanations(self):
        train_data = self.get_training_data()
        test_data = self.get_explainees()
        model = self.get_model()
        feature_names = self.get_feature_names(train_data[0])
        # background = train_data[np.random.choice(train_data.shape[0], 100, replace=False)]# we use the first 100 training examples as our background dataset to integrate over
        # explainer = shap.KernelExplainer(self.model.predict,  train_data, link="logit")
        # shap_values = explainer.shap_values(test_data, nsamples=100)
        # counter = 0
        # create_directory(ROOT_DIRECTORY +  'explanations')
        sess = K.get_session()
        explainer = shap.DeepExplainer(model, train_data[:100], sess)
        y_pred = model.predict(np.array([test_data[3]]))
        print(y_pred)
        shap_values = explainer.shap_values(np.array([test_data[2]]))
        # print('exp', explainer.expected_value)
        # print('input1', explainer.expected_value[0])
        # print('input2', shap_values[0][0])
        # print('input2a', shap_values[1])
        #shap.force_plot(explainer.expected_value[0], shap_values[0][0], show=False).savefig('demo.jpg')
        shap.multioutput_decision_plot
        shap.save_html('test.html', shap.force_plot(explainer.expected_value[1], shap_values[1][0], show=False, features=test_data[3]) )
        #shap.save_html('test.html', shap.summary_plot(shap_values[0][0], test_data[:1]) )
        counter = 0
        for household in test_data:
            if counter < 10:
                label = self.get_label(household)
                print(household.shape)
                shap_values = explainer.shap_values(np.array([household]))
                print(shap_values)
                shap.save_html('explanation_shap_' + str(counter) +  '.html', shap.force_plot(explainer.expected_value[self.get_label(household)], shap_values[self.get_label(household)][0], show=False, features=household))
            counter = counter + 1

