import pandas as pd
import numpy as np
from movie_dataset import movie_dataset_processing
from movie_lens import movie_lens_processing, clean_movie_title, build_mapping_table


def convertToTensors(user_matrix_rating):
    num_users = user_matrix_rating.shape[0]
    num_movies = user_matrix_rating.shape[1]
    tensor_3d = np.zeros((num_users, 1, num_movies))

    for i, user_row in enumerate(user_matrix_rating.values):
        tensor_3d[i, 0, :] = user_row

    return tensor_3d

def convertToTensorFromCombinedDataframes(df1, df2):
    array_t0 = df1.values
    array_t1 = df2.values

    assert array_t0.shape == array_t1.shape, "Los DataFrames deben tener la misma forma"

    tensor = np.stack([array_t0, array_t1], axis=0)
    return tensor

def preprocess():
    df_movie_dataset = pd.read_csv(r"../data/movie_dataset.csv")
    df_movies = pd.read_csv(r"../data/movielens/movie.csv")
    df_rating = pd.read_csv(r"../data/movielens/rating.csv")

    clean_movie_title(df_movies)
    mapping_table = build_mapping_table(df_movie_dataset, df_movies)

    movie_dataset_distance_matrix = movie_dataset_processing(df_movie_dataset, mapping_table)
    sim_matrix, matrix_ratings, user_matrix_rating = movie_lens_processing(df_rating, mapping_table)
    tensor_user_movie_rating = convertToTensors(user_matrix_rating)
    tensor_distances = convertToTensorFromCombinedDataframes(movie_dataset_distance_matrix, sim_matrix)
    return movie_dataset_distance_matrix, sim_matrix, matrix_ratings, tensor_user_movie_rating


if __name__ == "__main__":
    preprocess()