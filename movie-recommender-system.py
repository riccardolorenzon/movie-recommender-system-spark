complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

import os

datasets_path = os.path.join('..', 'datasets')

complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')

from urllib.request import urlretrieve

small_f = urlretrieve (small_dataset_url, small_dataset_path)
complete_f = urlretrieve (complete_dataset_url, complete_dataset_path)
