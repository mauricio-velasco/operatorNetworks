import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from movie_lens import clean_movie_title, build_mapping_table, remove_movies_without_rating
import pdb

def combined_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

def get_index_from_title(df, title):
    return df[df.title == title]["index"].values[0]

def get_index_from_mapping(df, mapping_table, title):
    df = df.merge(mapping_table, on='index')
    return df[df.title == title]["index"].values[0]


def get_title_from_index(df, index):
    return df[df.index == index]["title"].values[0]

def get_title_from_mapping(df, mapping_table, index):
    df = df.merge(mapping_table, on='index')
    return df[df.index == index]["title"].values[0]



def movie_dataset_processing(df, mapping_table):
    # Filter the interesting columns, rest filled with ''
    features = ['keywords', 'cast', 'genres', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')

    df = df.merge(mapping_table, on='index')[['new_id', 'keywords', 'cast', 'genres', 'director']].drop_duplicates()

    # New column with important features combined by a blank space.
    df["combined_features"] = df.apply(combined_features, axis=1)

    df.set_index('new_id', inplace=True)

    # Extract features
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])

    # Cosine similarity
    cosine_sim = cosine_similarity(count_matrix)

    ixs = df.index
    cosine_sim_df = pd.DataFrame(cosine_sim, index=ixs, columns=ixs)

    # Sort by index in rows and columns
    cosine_sim_df = cosine_sim_df.sort_index().sort_index(axis=1)

    return cosine_sim_df


if __name__ == '__main__':
    df = pd.read_csv(r"../data/movie_dataset.csv")
    df_movies = pd.read_csv(r"../data/movielens/movie.csv")
    df_rating = pd.read_csv(r"../data/movielens/rating.csv")

    clean_movie_title(df_movies)
    mapping_table = build_mapping_table(df, df_movies)
    mapping_table = remove_movies_without_rating(mapping_table, df_rating)

    distance_matrix = movie_dataset_processing(df, mapping_table)
    print("Finish successfully!")