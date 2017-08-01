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

# initiate a SparkConext
from pyspark.context import SparkContext
sc = SparkContext('local', 'movie-recommender-engine')

# give small_ratings_file in input to the sc
small_ratings_raw_data = sc.textFile(small_ratings_file)
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

# Parse the Raw data into a new RDD - Ratings.
data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')

small_movies_raw_data = sc.textFile(small_movies_file)
small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

# Parse the Raw data into a new RDD - Movies.
small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()
