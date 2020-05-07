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

## Reading the data
data = pd.read_csv(ROOT_DIRECTORY + "data/data_by_properties/single/single_train", delimiter=',')
data.replace([np.inf, -np.inf], np.nan)
np.nan_to_num(data)

data_frame = pd.DataFrame()

## Feature Category: Consumption

# Returns average consumption for a given time of the day and a given time of the week
def get_average_consumption_part_of_day(daytime_start, daytime_end, weekday_start, weekday_end):
    column = pd.DataFrame()
    for weekday in range (weekday_start, weekday_end):
       column[weekday] = data.iloc[:, daytime_start + (48 * weekday): daytime_end + (48 * weekday)].mean(axis=1)
    
    return column.mean(axis =1).tolist()

data_frame['c_week'] = get_average_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, MONDAY, SUNDAY)
data_frame['c_morning'] = get_average_consumption_part_of_day(FIRST_MORNING_START, FIRST_MORNING_END, MONDAY, SUNDAY)
data_frame['c_noon'] = get_average_consumption_part_of_day(FIRST_NOON_START, FIRST_NOON_END, MONDAY, SUNDAY)
data_frame['c_afternoon'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, SUNDAY)
data_frame['c_evening'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, SUNDAY)
data_frame['c_night'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, SUNDAY)

data_frame['c_weekday'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, FRIDAY)
data_frame['c_wd_morning'] = get_average_consumption_part_of_day(FIRST_MORNING_START, FIRST_MORNING_END, MONDAY, FRIDAY)
data_frame['c_wd_noon'] = get_average_consumption_part_of_day(FIRST_NOON_START, FIRST_NOON_END, MONDAY, FRIDAY)
data_frame['c_wd_afternoon'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, FRIDAY)
data_frame['c_wd_evening'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, FRIDAY)
data_frame['c_wd_night'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, MONDAY, FRIDAY)

data_frame['c_weekend'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, SATURDAY, SUNDAY)
data_frame['c_we_morning'] = get_average_consumption_part_of_day(FIRST_MORNING_START, FIRST_MORNING_END, SATURDAY, SUNDAY)
data_frame['c_we_noon'] = get_average_consumption_part_of_day(FIRST_NOON_START, FIRST_NOON_END, SATURDAY, SUNDAY)
data_frame['c_we_afternoon'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, SATURDAY, SUNDAY)
data_frame['c_we_evening'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, SATURDAY, SUNDAY)
data_frame['c_we_night'] = get_average_consumption_part_of_day(FIRST_EVENING_START, FIRST_EVENING_END, SATURDAY, SUNDAY)


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

data_frame['r_mean_max'] = get_ratio(data_frame['c_week'], temp_data_frame['c_max'])
data_frame['r_min_mean'] = get_ratio(temp_data_frame['c_min'], data_frame['c_week'])
data_frame['r_night_day'] = get_ratio(data_frame['c_night'], data_frame['c_week'])
data_frame['r_morning_noon'] = get_ratio(data_frame['c_night'], data_frame['c_week'])
data_frame['r_evening_noon'] = get_ratio(data_frame['c_morning'], data_frame['c_noon'])
data_frame['r_var_wd_we'] = get_ratio(temp_data_frame['c_var_weekday'], temp_data_frame['c_var_weekend'])
#data_frame['r_min_wd_we'] = get_ratio(temp_data_frame['c_min_weekday'], temp_data_frame['c_min_weekend'])
# data_frame['r_max_wd_we'] = get_ratio(temp_data_frame['c_max_weekday'], temp_data_frame['c_max_weekend'])
# data_frame['r_evening_wd_we'] = get_ratio(data_frame['c_wd_evening'], data_frame['c_we_evening'])
# data_frame['r_night_wd_we'] = get_ratio(data_frame['c_wd_night'], data_frame['c_we_night'])
# data_frame['r_noon_wd_we'] = get_ratio(data_frame['c_wd_noon'], data_frame['c_we_noon'])
# data_frame['r_morning_wd_we'] = get_ratio(data_frame['c_wd_morning'], data_frame['c_we_morning'])
# data_frame['r_afternoon_wd_we'] = get_ratio(data_frame['c_wd_afternoon'], data_frame['c_we_afternoon'])
data_frame['r_we_night_day'] = get_ratio(data_frame['c_we_night'], temp_data_frame['c_we_day'])
data_frame['r_we_morning_noon'] = get_ratio(data_frame['c_we_morning'], data_frame['c_we_noon'])
data_frame['r_we_evening_noon'] = get_ratio(data_frame['c_we_evening'], data_frame['c_we_noon'])
data_frame['r_wd_night_day'] = get_ratio(data_frame['c_wd_night'], temp_data_frame['c_wd_day'])
data_frame['r_wd_morning_noon'] = get_ratio(data_frame['c_wd_morning'], data_frame['c_wd_noon'])
data_frame['r_wd_evening_noon'] = get_ratio(data_frame['c_wd_evening'], data_frame['c_wd_noon'])

## Feature Category: Statistical Figures

def get_average_max_consumption_part_of_day(daytime_start, daytime_end, weekday_start, weekday_end):
    column = pd.DataFrame()
    for weekday in range (weekday_start, weekday_end):
       column[weekday] = data.iloc[:, daytime_start + (48 * weekday): daytime_end + (48 * weekday)].max(axis=1)
    
    return column.mean(axis =1).tolist()

def get_average_min_consumption_part_of_day(daytime_start, daytime_end, weekday_start, weekday_end):
    column = pd.DataFrame()
    for weekday in range (weekday_start, weekday_end):
       column[weekday] = data.iloc[:, daytime_start + (48 * weekday): daytime_end + (48 * weekday)].max(axis=1)
    
    return column.mean(axis =1).tolist()


pd.concat([data_frame, temp_data_frame])
data_frame['s_variance'] = get_variance_consumption(MONDAY, SUNDAY)
# data_frame['s_q1'] = data.iloc[:, 48 * MONDAY : 48 * SUNDAY].quantile(0.25)
# data_frame['s_q2'] = data.iloc[:, 48 * MONDAY : 48 * SUNDAY].quantile(0.5)
# data_frame['s_q2'] = data.iloc[:, 48 * MONDAY : 48 * SUNDAY].quantile(0.75)
data_frame['c_max_avg'] = get_average_max_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, MONDAY, SUNDAY)
data_frame['c_min_avg'] = get_average_min_consumption_part_of_day(FIRST_DAY_START, FIRST_DAY_END, MONDAY, SUNDAY)

print(data_frame.iloc[:, 31:40])




# ### Random Forrest
data_frame.drop(index=24, axis='columns')
print(np.where(np.isposinf(data_frame)))
np.nan_to_num(data_frame)
print(data_frame.iloc[56, 24])
train_features, test_features, train_labels, test_labels = train_test_split(data_frame, data.iloc[:, 0], test_size = 0.25, random_state = 42)
print(np.where(np.isnan(test_features)))
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:',train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(train_features,train_labels)

y_pred=clf.predict(test_features)
print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))