import numpy as np
from movieDataExample import raw_data_loader
import pandas as pd
import pdb
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationSystem:
    #This class holds the data for a recommendation system.
    # This data consists of user, item pairs
    # The structure remembers the user and item indices in the ORIGINAL DB
    # and places the interaction data by users in
    #       self.interactions_dict_by_user 
        
    def __init__(self,pandas_df_interactions, user_column_name, item_column_name,rating_column_name,rating_lower_bound, rating_upper_bound):
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
        # is called: self. interactions_dict_by_user
        self.interactions_dict_by_user = dict()
        for user_idx in self.userIdxs:
            reviewed_items_df = df_interactions[df_interactions[user_column_name]==user_idx]
            assert len(reviewed_items_df) == reviewed_items_df[item_column_name].nunique(),"repeated reviewing of a single item not allowed"
            reviewed_items_df = reviewed_items_df[[item_column_name,rating_column_name]]
            reviewed_items_df = reviewed_items_df.set_index(item_column_name)
            new_dict_entry = reviewed_items_df.to_dict()
            new_dict_entry = new_dict_entry[rating_column_name]
            assert all([rating_lower_bound<= j for j in new_dict_entry.values()]), "values below allowed lower bound"
            assert all([j<= rating_upper_bound for j in new_dict_entry.values()]), "values above allowed upper bound"
            assert set(new_dict_entry.keys()) <= set(self.itemIdxs), "Records improperly read, every rating must correspond to a known index"
            self.interactions_dict_by_user[user_idx] = new_dict_entry.copy()
        #TODO: build self.interactions_dict_by_item


    def user_idx_from_DB_user_id(self, DB_userId):
        #given a DATABASE user id compute the corresponding user_Idx
        return self.DB_userIds_list.index(DB_userId)

    def item_idx_from_DB_item_id(self, DB_itemId):
        #given a DATABASE user id compute the corresponding user_Idx
        return self.DB_itemIds_list.index(DB_itemId)


    def items_and_ratings_by_single_user(self, user_idx, recentered):
        #Returns the item_idx's list of items that have been rated by user_idx
        #Returns a dictionary with keys: item_idxs and values the ratings.
        if recentered == False:
            return self.interactions_dict_by_user[user_idx]

    def items_rated_by_pair_of_users(self, user_idx_pair):
        u = user_idx_pair[0]
        v = user_idx_pair[1]
        rated_items_set_u = set(self.interactions_dict_by_user[u].keys())
        rated_items_set_v = set(self.interactions_dict_by_user[v].keys())
        return rated_items_set_u & rated_items_set_v

    def items_rated_by_all_in_list_of_users(self, list_user_idx):
        itemSets_list = [] 
        for user_idx in list_user_idx:       
            rated_items_set = set(self.interactions_dict_by_user[user_idx].keys())
            itemSets_list.append(rated_items_set)    
        return set.intersection(*itemSets_list)

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

    def compute_Sigma_and_B_users_matrices(self):
        #Computes the sparse correlation matrix correctly
        nU = self.num_users
        self.Sigma_users_matrix = np.zeros([nU,nU])
        self.B_users_matrix = np.zeros([nU,nU])
        for id_u, u in enumerate(self.userIdxs):
            for id_v, v in enumerate(self.userIdxs):
                usersPair = [u,v]
                itemsS = self.items_rated_by_pair_of_users(usersPair)
                l = len(itemsS)
                sum = 0
                if l!=0:
                    ratings_u = self.interactions_dict_by_user[u]
                    ratings_v = self.interactions_dict_by_user[v]
                    mu = np.mean([ratings_u[j] for j in itemsS])
                    mv = np.mean([ratings_v[j] for j in itemsS])
                    ratings_u_centered = [ratings_u[j]-mu for j in itemsS]
                    ratings_v_centered = [ratings_v[j]-mv for j in itemsS]
                    self.Sigma_users_matrix[id_u,id_v] = np.vdot(ratings_u_centered,ratings_v_centered)
                    sigma_u = np.sqrt(np.vdot(ratings_u_centered,ratings_u_centered))
                    sigma_v = np.sqrt(np.vdot(ratings_v_centered,ratings_v_centered))
                    self.B_users_matrix[id_u,id_v] = self.Sigma_users_matrix[id_u,id_v]/(sigma_u*sigma_v)

        #Finally we subtract the identity
        for i in range(self.num_users):
            self.B_users_matrix[i,i] -= 1.0

    def compute_shift_operator_users(self, k_most_correlated):
        #Given an integer k_most_correlated it selects the part of the Pearson correlation B_matrix
        #with the most positively correlated users for each user and normalizes it so it becomes 
        #right stochastic (rows summing to one)
        assert self.num_users >= k_most_correlated, "The number of users must exceed the sparsification threshold index k"
        assert hasattr(self, "B_users_matrix"), "The correlation matrix must be computed first"
        B = self.B_users_matrix
        nU = self.num_users
        shift_operator_users_matrix = np.zeros([nU,nU])
        #Ma
        Large_Correlation_indices = np.argpartition(B,-k_most_correlated,axis=1)[:,-k_most_correlated:]
        #We keep only the large correlations
        for i in range(nU):
            for j in Large_Correlation_indices[i,:]:
                shift_operator_users_matrix[i,j] = B[i,j]        
        row_sums = shift_operator_users_matrix.sum(axis=1)
        shift_operator_users_matrix = shift_operator_users_matrix / row_sums[:, np.newaxis]
        self.shift_operator_users_matrix = shift_operator_users_matrix




if __name__ == "__main__":
    #We test the class with the movieLens dataset
    import os
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    print(os.chdir(absolute_path)) #Now we are inside the /movie_data_process dir
    df_movies, df_interactions = raw_data_loader(ubound_popular_movies=200, ubound_popular_users=50)

    #We have to construct an instance of the recommender class
    RS = RecommendationSystem(
        pandas_df_interactions = df_interactions, 
        user_column_name = "userId", 
        item_column_name = "movieId",
        rating_column_name = "rating",
        rating_lower_bound = 0.0,
        rating_upper_bound = 5.0
        )
    RS.compute_Sigma_and_B_users_matrices()
    RS.compute_shift_operator_users(15)

    ax = sns.heatmap(RS.B_users_matrix, linewidth=0.5)
    plt.show()
    ax = sns.heatmap(RS.shift_operator_users_matrix, linewidth=0.5)
    plt.show()

    print("Done!")


