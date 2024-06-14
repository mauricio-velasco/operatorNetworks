import numpy as np
from movieDataExample import raw_data_loader
import pandas as pd
import pdb


class RecommendationSystem:
    #This class holds the data for a recommendation system.
    # This data consists of user, item pairs
    # The user and item identities are consecutive integers user_idx, item_idx
    # The structure remembers the user and item indices in the ORIGINAL DB as well
    def __init__(self,pandas_df_interactions, user_column_name, item_column_name,rating_column_name):
        self.pandas_df_interactions = pandas_df_interactions
        self.userIndicesInDB = []
        self.itemIndicesInDB = []
 
        df_interactions = self.pandas_df_interactions #shortening for nicer pandas code below
        #We extract the vectors of indices of users and items IN THE ORIGINAL DB. We will use these indices throughout        
        userIds_np_array = df_interactions[user_column_name].unique()
        self.userIdxs = userIds_np_array.tolist()#in the original DB
        itemIds_np_array = df_interactions[item_column_name].unique()
        self.itemIdxs = itemIds_np_array.tolist()#in the original DB

        #Finally updates the counts of users and items
        self.num_users = len(self.userIdxs)
        self.num_items = len(self.itemIdxs)

        #Finally the data is organized in a dict of dicts to preserve sparcity
        self.interactions_dict_by_user = dict()
        for user_idx in self.userIdxs:
            reviewed_items_df = df_interactions[df_interactions[user_column_name]==user_idx]
            assert len(reviewed_items_df) == reviewed_items_df[item_column_name].nunique(),"repeated reviewing of a single item"
            reviewed_items_df = reviewed_items_df[[item_column_name,rating_column_name]]
            reviewed_items_df = reviewed_items_df.set_index(item_column_name)
            new_dict_entry = reviewed_items_df.to_dict()
            new_dict_entry = new_dict_entry[rating_column_name]
            #TODO: verify assert new_dict_entry.keys() in self.itemIdxs
            self.interactions_dict_by_user[user_idx] = new_dict_entry.copy()

    def user_idx_from_DB_user_id(self, DB_userId):
        #given a DATABASE user id compute the corresponding user_Idx
        return self.DB_userIds_list.index(DB_userId)

    def item_idx_from_DB_item_id(self, DB_itemId):
        #given a DATABASE user id compute the corresponding user_Idx
        return self.DB_itemIds_list.index(DB_itemId)


    def items_rated_by(self, user_idx, recentered):
        #Returns the item_idx's list of items that have been rated by user_idx
        #Returns a dictionary with keys: item_idxs and values the ratings.
        pass

    def users_who_rated(self, item_idx, recentered):
        #Returns the user_idx's who have rated item_idx
        #Returns a dictionary with keys: user_idxs and values the ratings.
        pass

    def compute_user_mean(self):
        #Sparse computation of all user means
        res = []
        for user_idx in self.userIdxs:
            rated_items_dict = self.items_rated_by(user_idx, recentered=False)
            res.append(np.mean(rated_items_dict.values))
        self.userMeans = res

    def compute_user_recentered_data(self):
        #Create the new dictionary user_recentered_data 
        #containing dictionaries with sparse RECENTERED data
        self.compute_user_mean()
        self.user_recentered_data = dict()
        for user_idx in self.userIdxs:
            rated_items_dict = self.items_rated_by(user_idx, recentered=False)
            mean_rating = self.userMeans[user_idx]
            centered_rated_items_dict = dict()
            for key, value in rated_items_dict.items():
                centered_rated_items_dict[key] = rated_items_dict[key]-mean_rating
            self.user_recentered_data[user_idx] = centered_rated_items_dict.copy()

if __name__ == "__main__":
    #We test the class with the movieLens dataset
    import os
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    print(os.chdir(absolute_path)) #Now we are inside the /movie_data_process dir
    df_movies, df_interactions = raw_data_loader(ubound_popular_movies=100, ubound_popular_users=20)

    #We have to construct an instance of the recommender class
    RS = RecommendationSystem(
        pandas_df_interactions = df_interactions, 
        user_column_name = "userId", 
        item_column_name = "movieId",
        rating_column_name = "rating"
        )
    pdb.set_trace()

    print("Done!")

        

