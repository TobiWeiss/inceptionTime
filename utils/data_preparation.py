from utils.utils import create_directory, read_csv_to_list
import csv
from utils.constants import PROPERTY_NAMES
from utils.constants import PREPARED_DATA_ROOT_DIRECTORY
from utils.constants import RESULTS_ROOT_DIRECTORY
from utils.constants import DATA_WEEKS_ROOT_DIRECORY
from utils.constants import DATA_PROPERTIES_ROOT_DIRECTORY
from utils.constants import TRAINING_TEST_DATA_RATIO
import re
import pandas as pd
import numpy as np
from scipy.stats.stats import zscore

def combine_consumption_property_data(consumption, properties):
    consumption_data_with_property = []
    
    for consumption_row in consumption:
        for property_row in properties:
            if int(consumption_row[0]) == int(property_row[0]) :
                consumption_data_with_property.append([property_row[1]] + consumption_row[1:-1])

    return consumption_data_with_property

def create_training_data(consumption_data_with_property, property):
    create_directory(PREPARED_DATA_ROOT_DIRECTORY +  property)
    with open(PREPARED_DATA_ROOT_DIRECTORY + property + '/' + property + '_train', 'w+') as myfile:
        wr = csv.writer(myfile)
        counter = 0
        amount_of_rows = len(consumption_data_with_property) * TRAINING_TEST_DATA_RATIO
        for row in consumption_data_with_property:
            if counter < amount_of_rows:
                wr.writerow(row)
                counter = counter + 1

def create_test_data(consumption_data_with_property, property):
    create_directory(PREPARED_DATA_ROOT_DIRECTORY +  property)
    with open(PREPARED_DATA_ROOT_DIRECTORY + property + '/' + property + '_test', 'w+') as myfile:
        wr = csv.writer(myfile)
        counter = int(len(consumption_data_with_property) * TRAINING_TEST_DATA_RATIO) + 1
        amount_of_rows = len(consumption_data_with_property)
        while counter < amount_of_rows:
            wr.writerow(consumption_data_with_property[counter])
            counter = counter + 1

def get_cooking_properties():
    survey = read_csv_to_list(DATA_PROPERTIES_ROOT_DIRECTORY + "properties.csv", ",")
    regex = re.compile('[a-zA-z]+ 4704:')
    question_indices = []
    for question in survey[0]:
        if regex.match(question):
                question_indices.append(survey[0].index(question))
    survey_as_df = pd.DataFrame(survey)
    cooking_properties_as_list = list()
    for index in range(len(survey_as_df.iloc[1:, 0])):
            cooking_properties_as_list.append([survey_as_df.iloc[index, 0], survey_as_df.iloc[index, question_indices[0]]])
    cooking_properties_as_list.pop(0)

    return cooking_properties_as_list

def get_water_heating_properpties():
    survey = read_csv_to_list(DATA_PROPERTIES_ROOT_DIRECTORY + "properties.csv", ",")
    survey.pop(0)
    water_heating_properties_as_list = list()
    for row in survey:
        if row[50] == '1' or row[51] == '1':
             water_heating_properties_as_list.append([row[0], 1])
        else:
            water_heating_properties_as_list.append([row[0], 0])

    return water_heating_properties_as_list

def get_space_heating_properties():
    survey = read_csv_to_list(DATA_PROPERTIES_ROOT_DIRECTORY + "properties.csv", ",")
    survey.pop(0)
    water_heating_properties_as_list = list()
    for row in survey:
        if row[41] == '1' or row[42] == '1':
             water_heating_properties_as_list.append([row[0], 1])
        else:
            water_heating_properties_as_list.append([row[0], 0])

    return water_heating_properties_as_list

    return

def get_devices_properties():
    survey = read_csv_to_list(DATA_PROPERTIES_ROOT_DIRECTORY + "properties.csv", ",")
    regex1 = re.compile('[a-zA-z]+ 49002:')
    regex2 = re.compile('[a-zA-z]+ 490002:')
    question_indices = []
    num_devices_as_list = list()
    for question in survey[0]:
        if regex1.match(question) or regex2.match(question):
                question_indices.append(survey[0].index(question))
    survey.pop(0)
    num_devices_per_household = list()
    for row in survey:
        num_devices = 0
        for index in question_indices:
            num_devices += int(row[index])
        num_devices_per_household.append(num_devices)
    
    lower_percentile = np.percentile(np.array(num_devices_per_household), 33)
    middle_percentile = np.percentile(np.array(num_devices_per_household), 66)

    survey_as_df = pd.DataFrame(survey)

    for index in range(len(num_devices_per_household)):
        if num_devices_per_household[index] <= lower_percentile:
            num_devices = 0
        elif num_devices_per_household[index] <= middle_percentile:
            num_devices = 1
        else:
            num_devices = 2
        
        num_devices_as_list.append([survey_as_df.iloc[index, 0], num_devices])

    return num_devices_as_list

def prepare_consumption_data(consumption_data):
    consumption_data_as_df = pd.DataFrame(consumption_data)
    consumption_data_as_df = consumption_data_as_df.dropna()
    consumption_data_as_df.iloc[:, 1:] = consumption_data_as_df.iloc[:, 1:].apply(zscore)
    consumption_data = consumption_data_as_df.values.tolist()
    print(consumption_data)


    return consumption_data

def separate_data_to_train_test(week, property):
    
    consumption = read_csv_to_list(DATA_WEEKS_ROOT_DIRECORY + "DateienWoche" + week)
    consumption_prepared = prepare_consumption_data(consumption)

    property_functions = {
        'cooking': get_cooking_properties(),
        'water_heating': get_water_heating_properpties(),
        'space_heating': get_space_heating_properties(),
        'num_devices': get_devices_properties()
    }

    properties = property_functions[property]

    consumption_data_with_property = combine_consumption_property_data(consumption_prepared, properties)
    
    create_training_data(consumption_data_with_property, property)
    create_test_data(consumption_data_with_property, property)