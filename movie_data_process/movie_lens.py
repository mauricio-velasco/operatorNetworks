import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def clean_movie_title(df_movies):
    # Limpio la columna title, quitando el año (que viene por defecto en todos los titles)
    df_movies['title'] = df_movies['title'].str.extract('(.+?) \(\d+\)')
    #return df_movies

def build_mapping_table(df_movie_dataset, df_movies):
    # Hago el join entre los dos data frames de películas por la columna title de cada uno.
    universe_movie_table = pd.merge(df_movies, df_movie_dataset, how="inner", on="title")
    # .fillna(0)
    # Agrego una columna "new_id" autonumerada (dado que id ya existía)
    universe_movie_table['new_id'] = range(0, len(universe_movie_table))

    filt_columns = ["new_id", "title", "movieId", "index"]
    mapping_index_movie_table = universe_movie_table[filt_columns]

    return mapping_index_movie_table


def movie_lens_processing(df_rating, mapping_index_movie_table):
    # Clean column title, removing year (it comes by default concatenated to title)
    #clean_movie_title(df_movies)
    #mapping_index_movie_table = build_mapping_table(df_movie_dataset, df_movies)

    filt_df_ratings = pd.merge(df_rating, mapping_index_movie_table, on="movieId")[
        ["userId", "movieId", "rating"]].drop_duplicates()

    filt_df_ratings_with_id = pd.merge(filt_df_ratings, mapping_index_movie_table, on="movieId")

    # Every value corresponding to the user not watching the movie is 0.
    # Each row is a movie. Each column is a user. Intersection is rating.
    matrix_ratings = filt_df_ratings_with_id.pivot(index='new_id', columns='userId', values='rating').fillna(0)

    # For Mauri's process
    # Each row is a user. Each column is a movie. Intersection is rating.
    user_matrix_rating = filt_df_ratings_with_id.pivot(index='userId', columns='new_id', values='rating').fillna(0)

    # Train test divide (80/20)
    #ratings_train, ratings_test = train_test_split(matrix_ratings, test_size=0.2, random_state=42)
    #rating_indexes = ratings_train.index
    rating_indexes = matrix_ratings.index

    sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(matrix_ratings)

    # New df with same indexes
    sim_matrix_df = pd.DataFrame(sim_matrix, index=rating_indexes, columns=rating_indexes)
    sim_matrix_df = sim_matrix_df.sort_index().sort_index(axis=1)

    return sim_matrix_df, matrix_ratings, user_matrix_rating


if __name__ == '__main__':
    df_movie_dataset = pd.read_csv(r"../data/movie_dataset.csv")
    df_movies = pd.read_csv(r"../data/movielens/movie.csv")
    df_rating = pd.read_csv(r"../data/movielens/rating.csv")

    clean_movie_title(df_movies)
    mapping_index_movie_table = build_mapping_table(df_movie_dataset, df_movies)
    sim_matrix, matrix_ratings, user_matrix_rating = movie_lens_processing(df_rating, mapping_index_movie_table)




