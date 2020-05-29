#Root directory
ROOT_DIRECTORY = '/Users/tobiasweiss/ma/InceptionTime/'

#Root directory for prepared data used for training
PREPARED_DATA_ROOT_DIRECTORY = 'data/prepared_data_by_properties/'

#Root directory for data by weeks
DATA_WEEKS_ROOT_DIRECORY = 'data/data_cer/'

#Root directory for encoded properties 
PROPERTIES_ENCODED_DIRECTORY = 'data/properties_encoded/'

#Root directory for properties by household_id
DATA_PROPERTIES_ROOT_DIRECTORY = 'data/data_cer/'

#Root direcotry for results
RESULTS_ROOT_DIRECTORY = 'results/'

#Ratio desribing how much of the data is used for training and testing
TRAINING_TEST_DATA_RATIO = 0.8

#Reference Properties you wish to classify here
PROPERTY_NAMES = ['cooking', 'water_heating', 'space_heating', 'num_devices']

#Weeks of the cer data set used
WEEKS = list(range(30, 51))