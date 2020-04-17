UNIVARIATE_DATASET_NAMES = ['CER']

# UNIVARIATE_DATASET_NAMES = ['Meat', 'Coffee']

#UNIVARIATE_ARCHIVE_NAMES = ['TSC', 'InlineSkateXPs', 'SITS']
UNIVARIATE_ARCHIVE_NAMES = ['TSC']

SITS_DATASETS = ['SatelliteFull_TRAIN_c301', 'SatelliteFull_TRAIN_c200', 'SatelliteFull_TRAIN_c451',
                 'SatelliteFull_TRAIN_c89', 'SatelliteFull_TRAIN_c677', 'SatelliteFull_TRAIN_c59',
                 'SatelliteFull_TRAIN_c133']

InlineSkateXPs_DATASETS = ['InlineSkate-32', 'InlineSkate-64', 'InlineSkate-128',
                           'InlineSkate-256', 'InlineSkate-512', 'InlineSkate-1024',
                           'InlineSkate-2048']

dataset_names_for_archive = {'TSC': UNIVARIATE_DATASET_NAMES,
                             'SITS': SITS_DATASETS,
                             'InlineSkateXPs': InlineSkateXPs_DATASETS}
