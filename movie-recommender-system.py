complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

import os
import six

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
small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')

small_movies_raw_data = sc.textFile(small_movies_file)
small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

# Parse the Raw data into a new RDD - Movies.
small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()

# Prepare RDDs for training/validation/test
training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0)

# Map validation and test RDDs
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

# train Alternative Least Squares model
from pyspark.mllib.recommendation import ALS
import math

seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print ('For rank {} the RMSE is {}'.format(rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank

print ('The best model was trained with rank {}'.format(best_rank))
