import pandas as pd
import pdb

def clean_movie_title(df_movies):
    # Limpio la columna title, quitando el año (que viene por defecto en todos los titles)
    df_movies['title'] = df_movies['title'].str.extract('(.+?) \(\d+\)')
    #return df_movies


def select_movies_and_users_w_most_data(ubound_popular_movies, ubound_popular_users, df_movies, df_interactions):
    #This function selects the data which refers to the most popular movies and the most frquent users.
    #The target amount of movies and users is given by the parameters ubound_popular_movies and ubound_popular_users 
    #It KEEPS the NA data as NA (it does NOT introduce zeroes or any other extra)

    #First, we create dataframes of counts which measure popularity of movies and users
    users_ranked = df_interactions.userId.value_counts()
    top_users_ranked = users_ranked.nlargest(ubound_popular_users)
    
    movies_ranked = df_interactions.movieId.value_counts()
    top_movies_ranked = movies_ranked.nlargest(ubound_popular_movies)

    #Then we select the ratings given by power users to power movies
    df_interactions = df_interactions[df_interactions.userId.isin(top_users_ranked.index)]
    df_interactions = df_interactions[df_interactions.movieId.isin(top_movies_ranked.index)]
    df_movies = df_movies[df_movies.movieId.isin(top_movies_ranked.index)]

    #And finally return the databases for our restricted universe,
    return df_movies, df_interactions



def raw_data_loader(ubound_popular_movies,ubound_popular_users):
    #This function loads the raw MovieLens data and restricts it to the most 
    #popular users and movies WITHOUT eliminating missing data

    #df_movie_dataset = pd.read_csv(r"../data/movie_dataset.csv")
    #df_movie_dataset: rows are movies and columns are: index, budget, genres, director,...

    df_movies = pd.read_csv(r"../data/movielens/movie.csv")
    #df_movies: rows are movies and the columns are movieId, title and genres
    df_interactions = pd.read_csv(r"../data/movielens/rating.csv")
    #df_interactions: rows are (user, item) interactions (i.e. someone watching a movie) 
    #and columns contain userId movieId and given ratings together with a timestamp
    clean_movie_title(df_movies)
    df_movies, df_interactions = select_movies_and_users_w_most_data(ubound_popular_movies, ubound_popular_users,  df_movies, df_interactions)
    return df_movies, df_interactions
    



if __name__ == "__main__":
    #We test the class with the movieLens dataset
    import os
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    print(os.chdir(absolute_path)) #Now we are inside the /Collaborative_Filtering_extension
    ubound_popular_movies = 100
    ubound_popular_users = 1000
    raw_data_loader(ubound_popular_movies,ubound_popular_users)
    print("Done!")
