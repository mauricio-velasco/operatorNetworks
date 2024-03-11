import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from movie_lens import clean_movie_title, build_mapping_table


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



def process_movie_dataset(df=pd.read_csv(r"data/movie_dataset.csv"), df_movies=pd.read_csv(r"data/movielens/movie.csv")):
    clean_movie_title(df_movies)
    mapping_table = build_mapping_table(df, df_movies)
    # Me quedo con las columnas que me importan y aquellos datos con Nan, los lleno con ''
    features = ['keywords', 'cast', 'genres', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')

    df = df.merge(mapping_table, on='index')[['new_id', 'keywords', 'cast', 'genres', 'director']].drop_duplicates()

    # Genero una columna con las features concatenadas de espacios
    df["combined_features"] = df.apply(combined_features, axis=1)

    df.set_index('new_id', inplace=True)

    # Extract features
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])

    #print("Count Matrix:", count_matrix.toarray())

    # Cosine similarity
    cosine_sim = cosine_similarity(count_matrix)

    ixs = df.index
    cosine_sim_df = pd.DataFrame(cosine_sim, index=ixs, columns=ixs)

    return cosine_sim_df




if __name__ == '__main__':
    df = pd.read_csv(r"data/movie_dataset.csv")
    df_movies = pd.read_csv(r"data/movielens/movie.csv")

    clean_movie_title(df_movies)
    #mapping_table = build_mapping_table(df, df_movies)
    distance_matrix = process_movie_dataset(df, df_movies)

    # Use example
    #movie_user_likes = "Dead Poets Society"
    #movie_index = get_index_from_title(df, movie_user_likes)
    #movie_index = get_index_from_mapping(df, mapping_table, movie_user_likes)

    #similar_movies = list(enumerate(distance_matrix[movie_index]))
    #sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    # Test
    #i = 0
    #for movie in sorted_similar_movies:
    #    print(get_title_from_mapping(df, movie[0]))
    #    i = i + 1
    #    if i > 15:
    #        break
