import pandas as pd
import numpy as np
from movie_dataset import movie_dataset_processing
from movie_lens import movie_lens_processing, clean_movie_title, build_mapping_table, remove_movies_without_rating
import pdb

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

def preprocess(ubound_popular_movies, ubound_popular_users):
    df_movie_dataset = pd.read_csv(r"../data/movie_dataset.csv")
    df_movies = pd.read_csv(r"../data/movielens/movie.csv")
    df_rating = pd.read_csv(r"../data/movielens/rating.csv")

    clean_movie_title(df_movies)
    
    df_rating, df_movie_dataset, df_movies = select_movies_and_users_w_data(ubound_popular_movies, ubound_popular_users, df_movie_dataset, df_movies, df_rating)
    #Compute mapping_table to our restricted universe
    mapping_table = build_mapping_table(df_movie_dataset, df_movies)
    mapping_table = remove_movies_without_rating(mapping_table, df_rating)

    movie_dataset_distance_matrix = movie_dataset_processing(df_movie_dataset, mapping_table)
    sim_matrix, matrix_ratings, user_matrix_rating = movie_lens_processing(df_rating, mapping_table)

    tensor_user_movie_rating = convertToTensors(user_matrix_rating)
    tensor_distances = convertToTensorFromCombinedDataframes(movie_dataset_distance_matrix, sim_matrix)
    #return movie_dataset_distance_matrix, sim_matrix, matrix_ratings, tensor_user_movie_rating
    return tensor_distances, tensor_user_movie_rating


def select_movies_and_users_w_data(ubound_popular_movies, ubound_popular_users, df_movie_dataset, df_movies, df_rating):
    #This function selects the data which refers to the most popular movies and the most frquent users.
    #The amount of movies and users is given by the parameters ubound_popular_movies and ubound_popular_users 

    #First, we create dataframes of counts which measure popularity of movies and users
    users_ranked = df_rating.userId.value_counts()
    users_ranked = users_ranked.nlargest(ubound_popular_users)
    
    movies_ranked = df_rating.movieId.value_counts()
    movies_ranked = movies_ranked.nlargest(ubound_popular_movies)

    #Then we select the ratings given by power users to power movies
    df_rating = df_rating[df_rating.userId.isin(users_ranked.index)]
    df_rating = df_rating[df_rating.movieId.isin(movies_ranked.index)]
    df_movies = df_movies[df_movies.movieId.isin(movies_ranked.index)]
    df_movie_dataset = df_movie_dataset[df_movie_dataset.title.isin(df_movies.title)]

    #And finally return the databases for our restricted universe,
    return df_rating, df_movie_dataset, df_movies


if __name__ == "__main__":
    import os
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    print(os.chdir(absolute_path)) #Now we are inside the /movie_data_process dir
    ubound_popular_movies = 100
    ubound_popular_users = 1000
    preprocess(ubound_popular_movies,ubound_popular_users)
    print("Finish!")