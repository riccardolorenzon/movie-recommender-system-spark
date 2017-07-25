complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

import os

datasets_path = os.path.join('..', 'datasets')

complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')

from urllib.request import urlretrieve

if not os.path.exists(small_dataset_path):
    small_f = urlretrieve (small_dataset_url, small_dataset_path)
if not os.path.exists(complete_dataset_path):
    complete_f = urlretrieve (complete_dataset_url, complete_dataset_path)

import zipfile

if not os.path.exists(small_dataset_path):
    with zipfile.ZipFile(small_dataset_path, "r") as z:
        z.extractall(datasets_path)

if not os.path.exists(complete_dataset_path):
    with zipfile.ZipFile(complete_dataset_path, "r") as z:
        z.extractall(datasets_path)

small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')

# initiate an RDD with the ratings raw file
from pyspark.context import SparkContext
sc = SparkContext('local', 'movie-recommender-engine')
small_ratings_raw_data = sc.textFile(small_ratings_file)
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]
