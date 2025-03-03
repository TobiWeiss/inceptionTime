from numpy import loadtxt
import os 
import sys
sys.path.insert(1, '../')
import pandas as pd
import numpy as np

from utils.constants import ROOT_DIRECTORY
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
import lime
import lime.lime_tabular

### Feature Extraction

## Constants

# Days of the week as integers
MONDAY = 0
TUESDAY = 1
WEDNESDAY = 2
THURSDAY = 3
FRIDAY = 4
SATURDAY = 5
SUNDAY = 6

# Indexes indicating where in the dataframe certain times of the day/week begin or end
# The first day starts at index 1 since index 0 contains the class of the observation
FIRST_DAY_START = 1
FIRST_DAY_END = 48
FIRST_MORNING_START = 13
FIRST_MORNING_END = 20
FIRST_NOON_START = 21
FIRST_NOON_END = 28
FIRST_AFTERNOON_START = 29
FIRST_AFTERNOON_END = 36
FIRST_EVENING_START = 37
FIRST_EVENING_END = 44
FIRST_NIGHT_START = 3
FIRST_NGIHT_END = 12

class RandomForrest:
    def __init__(self, property_name):
        self.property_name = property_name
        self.labels = list()
        self.data_frame = pd.DataFrame()
    
    def prepare_data(self):
        ## Reading the data
        data = pd.read_csv(ROOT_DIRECTORY + "data/prepared_data_by_properties/" + self.property_name + '/' + self.property_name + '_train', delimiter=',')
        self.labels = data.iloc[:, 0]

        ## Feature Category: Consumption

        # Returns average consumption for a given time of the day and a given time of the week
        def get_average_consumption_part_of_day(daytime_start, daytime_end, weekday_start, weekday_end):
            column = pd.DataFrame()
            for weekday in range (weekday_start, weekday_end):
                column[weekday] = data.iloc[:, daytime_start + (48 * weekday): daytime_end + (48 * weekday)].mean(axis=1)
    
            return column.mean(axis =1).tolist()

        self.data_frame['c_week'] = get_average_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, MONDAY, SUNDAY)
        self.data_frame['c_morning'] = get_average_consumption_part_of_day(FIRST_MORNING_START, FIRST_MORNING_END, MONDAY, SUNDAY)
        self.data_frame['c_noon'] = get_average_consumption_part_of_day(FIRST_NOON_START, FIRST_NOON_END, MONDAY, SUNDAY)
        self.data_frame['c_afternoon'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, SUNDAY)
        self.data_frame['c_evening'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, SUNDAY)
        self.data_frame['c_night'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, SUNDAY)

        self.data_frame['c_weekday'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, FRIDAY)
        self.data_frame['c_wd_morning'] = get_average_consumption_part_of_day(FIRST_MORNING_START, FIRST_MORNING_END, MONDAY, FRIDAY)
        self.data_frame['c_wd_noon'] = get_average_consumption_part_of_day(FIRST_NOON_START, FIRST_NOON_END, MONDAY, FRIDAY)
        self.data_frame['c_wd_afternoon'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, FRIDAY)
        self.data_frame['c_wd_evening'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, FRIDAY)
        self.data_frame['c_wd_night'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, FRIDAY)

        self.data_frame['c_weekend'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, SATURDAY, SUNDAY)
        self.data_frame['c_we_morning'] = get_average_consumption_part_of_day(FIRST_MORNING_START, FIRST_MORNING_END, SATURDAY, SUNDAY)
        self.data_frame['c_we_noon'] = get_average_consumption_part_of_day(FIRST_NOON_START, FIRST_NOON_END, SATURDAY, SUNDAY)
        self.data_frame['c_we_afternoon'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, SATURDAY, SUNDAY)
        self.data_frame['c_we_evening'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, SATURDAY, SUNDAY)
        self.data_frame['c_we_night'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, SATURDAY, SUNDAY)


        ## Feature Category: Relations

        # Returns maximum consumption for a given time of the day and a given time of the week
        def get_max_consumption_part_of_day(daytime_start, daytime_end, weekday_start, weekday_end):
            column = pd.DataFrame()
            for weekday in range (weekday_start, weekday_end):
                column[weekday] = data.iloc[:, daytime_start + (48 * weekday): daytime_end + (48 * weekday)].max(axis=1)
    
            return column.max(axis =1).tolist()

        # Returns minimum consumption for a given time of the day and a given time of the week
        def get_min_consumption_part_of_day(daytime_start, daytime_end, weekday_start, weekday_end):
            column = pd.DataFrame()
            for weekday in range (weekday_start, weekday_end):
                column[weekday] = data.iloc[:, daytime_start + (48 * weekday): daytime_end + (48 * weekday)].min(axis=1)
    
            return column.min(axis =1).tolist()

        # Returns variance of consumption for a given time of the day and a given time of the week
        def get_variance_consumption(weekday_start, weekday_end):
            column = pd.DataFrame()
            consumption = data.iloc[:, 48 * weekday_start : 48 * weekday_end]
    
            return consumption.var(axis =1).tolist()

        def get_ratio(first_column, second_column):
            return first_column / second_column


        temp_data_frame = pd.DataFrame()

        temp_data_frame['c_max'] = get_max_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, MONDAY, SUNDAY)
        temp_data_frame['c_min'] = get_min_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, MONDAY, SUNDAY)
        temp_data_frame['c_max_weekday'] = get_max_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, MONDAY, FRIDAY)
        temp_data_frame['c_min_weekday'] = get_min_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, MONDAY, FRIDAY)
        temp_data_frame['c_max_weekend'] = get_max_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, SATURDAY, SUNDAY)
        temp_data_frame['c_min_weekend'] = get_min_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, SATURDAY, SUNDAY)
        temp_data_frame['c_var_weekday'] = get_variance_consumption(MONDAY, FRIDAY)
        temp_data_frame['c_var_weekend'] = get_variance_consumption(SATURDAY, SUNDAY)
        temp_data_frame['c_wd_day'] = get_average_consumption_part_of_day(FIRST_MORNING_START, FIRST_AFTERNOON_END, MONDAY, FRIDAY)
        temp_data_frame['c_we_day'] = get_average_consumption_part_of_day(FIRST_MORNING_START, FIRST_AFTERNOON_END, SATURDAY, SUNDAY)

        self.data_frame['r_mean_max'] = get_ratio(self.data_frame['c_week'], temp_data_frame['c_max'])
        self.data_frame['r_min_mean'] = get_ratio(temp_data_frame['c_min'], self.data_frame['c_week'])
        self.data_frame['r_night_day'] = get_ratio(self.data_frame['c_night'], self.data_frame['c_week'])
        self.data_frame['r_morning_noon'] = get_ratio(self.data_frame['c_night'], self.data_frame['c_week'])
        self.data_frame['r_evening_noon'] = get_ratio(self.data_frame['c_morning'], self.data_frame['c_noon'])
        self.data_frame['r_var_wd_we'] = get_ratio(temp_data_frame['c_var_weekday'], temp_data_frame['c_var_weekend'])
        self.data_frame['r_min_wd_we'] = get_ratio(temp_data_frame['c_min_weekday'], temp_data_frame['c_min_weekend'])
        self.data_frame['r_max_wd_we'] = get_ratio(temp_data_frame['c_max_weekday'], temp_data_frame['c_max_weekend'])
        self.data_frame['r_evening_wd_we'] = get_ratio(self.data_frame['c_wd_evening'], self.data_frame['c_we_evening'])
        self.data_frame['r_night_wd_we'] = get_ratio(self.data_frame['c_wd_night'], self.data_frame['c_we_night'])
        self.data_frame['r_noon_wd_we'] = get_ratio(self.data_frame['c_wd_noon'], self.data_frame['c_we_noon'])
        self.data_frame['r_morning_wd_we'] = get_ratio(self.data_frame['c_wd_morning'], self.data_frame['c_we_morning'])
        self.data_frame['r_afternoon_wd_we'] = get_ratio(self.data_frame['c_wd_afternoon'], self.data_frame['c_we_afternoon'])
        self.data_frame['r_we_night_day'] = get_ratio(self.data_frame['c_we_night'], temp_data_frame['c_we_day'])
        self.data_frame['r_we_morning_noon'] = get_ratio(self.data_frame['c_we_morning'], self.data_frame['c_we_noon'])
        self.data_frame['r_we_evening_noon'] = get_ratio(self.data_frame['c_we_evening'], self.data_frame['c_we_noon'])
        self.data_frame['r_wd_night_day'] = get_ratio(self.data_frame['c_wd_night'], temp_data_frame['c_wd_day'])
        self.data_frame['r_wd_morning_noon'] = get_ratio(self.data_frame['c_wd_morning'], self.data_frame['c_wd_noon'])
        self.data_frame['r_wd_evening_noon'] = get_ratio(self.data_frame['c_wd_evening'], self.data_frame['c_wd_noon'])

        ## Feature Category: Statistical Figures

        def get_average_max_consumption_part_of_day(daytime_start, daytime_end, weekday_start, weekday_end):
            column = pd.DataFrame()
            for weekday in range (weekday_start, weekday_end):
                column[weekday] = data.iloc[:, daytime_start + (48 * weekday): daytime_end + (48 * weekday)].max(axis=1)
    
            return column.mean(axis =1).tolist()

        def get_average_min_consumption_part_of_day(daytime_start, daytime_end, weekday_start, weekday_end):
            column = pd.DataFrame()
            for weekday in range (weekday_start, weekday_end):
                column[weekday] = data.iloc[:, daytime_start + (48 * weekday): daytime_end + (48 * weekday)].min(axis=1)
    
            return column.mean(axis =1).tolist()


        pd.concat([self.data_frame, temp_data_frame])
        self.data_frame['s_variance'] = get_variance_consumption(MONDAY, SUNDAY)
        # self.data_frame['s_q1'] = data.iloc[:, 48 * MONDAY : 48 * SUNDAY].quantile(0.25)
        # self.data_frame['s_q2'] = data.iloc[:, 48 * MONDAY : 48 * SUNDAY].quantile(0.5)
        # self.data_frame['s_q2'] = data.iloc[:, 48 * MONDAY : 48 * SUNDAY].quantile(0.75)
        self.data_frame['c_max_avg'] = get_average_max_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, MONDAY, SUNDAY)
        self.data_frame['c_min_avg'] = get_average_min_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, MONDAY, SUNDAY)


# ### Random Forrest
    def classify(self):
        #self.data_frame.drop(index=24, axis='columns')
        train_features, test_features, train_labels, test_labels = train_test_split(self.data_frame, self.labels, test_size = 0.25, random_state = 42)

        #Create a Gaussian Classifier
        clf=RandomForestClassifier(n_estimators=100)

        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(train_features,train_labels)

        y_pred=clf.predict(test_features)
        print("Accuracy ( " + self.property_name + " ):",metrics.accuracy_score(test_labels, y_pred))
        print("MCC ( " + self.property_name + " ):",metrics.matthews_corrcoef(test_labels, y_pred))
        #print("Precision:",metrics.precision_score(test_labels, y_pred))
        #print("Recall:",metrics.recall_score(test_labels, y_pred))
        self.explain(train_features, test_features, clf)
    
    def explain(self, train_features, test_features, clf):
        feature_names = list(train_features.columns)
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(train_features), feature_names=feature_names, class_names=['not single', 'single'], discretize_continuous=True)
        exp = explainer.explain_instance(np.array(test_features.iloc[2]), clf.predict_proba, num_features=10, top_labels=0)
        exp.save_to_file("explanations/explanation_" +  'rf' + "_" + self.property_name + ".html")
