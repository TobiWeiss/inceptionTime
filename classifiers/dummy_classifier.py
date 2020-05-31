import pandas as pd
from utils.constants import ROOT_DIRECTORY
from sklearn.dummy import DummyClassifier

class DummyClf: 
    def __init__(self, property_name):
        self.property_name = property_name
        self.labels = list()
        self.data_frame = pd.DataFrame()
    
    def classify(self):
        data = pd.read_csv(ROOT_DIRECTORY + "data/prepared_data_by_properties/" + self.property_name + '/' + self.property_name + '_train', delimiter=',')
        self.labels = data.iloc[:, 0]
        self.data_frame = data.iloc[:, 1:]
        dummy_clf = DummyClassifier(strategy="stratified")
        dummy_clf.fit(self.data_frame, self.labels)
        dummy_clf.predict(self.data_frame)
        print(dummy_clf.score(self.data_frame, self.labels))
        