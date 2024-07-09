import torch
import numpy as np
#from movieDataExample import raw_data_loader
import pandas as pd
import pdb
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random

class RecommendationSystem:
    #This class holds the data for a recommendation system.
    # This data consists of user, item pairs
    # The structure remembers the user and item indices in the ORIGINAL DB
    # and places the interaction data by users in
    #       self.interactions_dict_by_user 
        
    def __init__(self, data_core):        
        #We begin by copying basic info from the data_core class which indexes the data
        self.userIdxs = data_core.userIdxs
        self.itemIdxs = data_core.itemIdxs
        self.df_interactions = data_core.df_interactions
        self.df_items = data_core.df_items
        user_column_name = data_core.user_column_name
        item_column_name = data_core.item_column_name
        rating_column_name = data_core.rating_column_name

        #updates the counts of users and items
        self.num_users = len(self.userIdxs)
        self.num_items = len(self.itemIdxs)
        
        #Finally the data is organized in a dict of dicts to preserve sparcity
        # is called: self. interactions_dict_by_user
        df_interactions = self.df_interactions
        self.interactions_dict_by_user = dict()
        for user_idx in self.userIdxs:
            reviewed_items_df = df_interactions[df_interactions[user_column_name]==user_idx]
            assert len(reviewed_items_df) == reviewed_items_df[item_column_name].nunique(),"repeated reviewing of a single item not allowed"
            reviewed_items_df = reviewed_items_df[[item_column_name,rating_column_name]]
            reviewed_items_df = reviewed_items_df.set_index(item_column_name)
            new_dict_entry = reviewed_items_df.to_dict()
            new_dict_entry = new_dict_entry[rating_column_name]
            assert set(new_dict_entry.keys()) <= set(self.itemIdxs), "Records improperly read, every rating must correspond to a known index"
            self.interactions_dict_by_user[user_idx] = new_dict_entry.copy()

        self.compute_user_recentered_data()
        #TODO: build self.interactions_dict_by_item

    def items_and_ratings_by_single_user(self, user_idx, recentered):
        #Returns the item_idx's list of items that have been rated by user_idx
        #Returns a dictionary with keys: item_idxs and values the ratings.
        if recentered == False:
            return self.interactions_dict_by_user[user_idx]
        if recentered == True:
            return self.recentered_interactions_dict_by_user[user_idx]

    def items_rated_by_pair_of_users(self, user_idx_pair):
        u = user_idx_pair[0]
        v = user_idx_pair[1]
        rated_items_set_u = set(self.interactions_dict_by_user[u].keys())
        rated_items_set_v = set(self.interactions_dict_by_user[v].keys())
        return rated_items_set_u & rated_items_set_v

    def compute_user_means(self):
        #Sparse computation of all user mean ratings
        res = []
        for user_idx in self.userIdxs:
            rated_items_dict = self.items_and_ratings_by_single_user(user_idx, recentered=False)
            res.append(np.array(list(rated_items_dict.values())).mean())
        self.userMeans = dict(zip(self.userIdxs, res))

    def compute_user_recentered_data(self):
        #Create the new dictionary user_recentered_data 
        #containing dictionaries with sparse RECENTERED data
        self.compute_user_means()
        self.recentered_interactions_dict_by_user = dict()
        for user_idx in self.userIdxs:
            rated_items_dict = self.items_and_ratings_by_single_user(user_idx, recentered=False)
            mean_rating = self.userMeans[user_idx]
            centered_rated_items_dict = dict()
            for key, value in rated_items_dict.items():
                centered_rated_items_dict[key] = rated_items_dict[key]-mean_rating
            self.recentered_interactions_dict_by_user[user_idx] = centered_rated_items_dict.copy()

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
                    if sigma_u*sigma_v != 0.0:
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
        
        shift_operator_users_matrix = torch.zeros([nU,nU])
        #Selects the largest entries?
        Large_Correlation_indices = np.argpartition(B,-k_most_correlated,axis=1)[:,-k_most_correlated:]
        #We keep only the large correlations
        for i in range(nU):
            for j in Large_Correlation_indices[i,:]:
                shift_operator_users_matrix[i,j] = B[i,j]        
        row_sums = shift_operator_users_matrix.sum(axis=1)
        shift_operator_users_matrix = shift_operator_users_matrix / row_sums[:, np.newaxis]        
        #We transpose the shift operator matrix so that it acts as a Markov chain when right-multiplied
        shift_operator_users_matrix = shift_operator_users_matrix.t()
        self.shift_operator_users_matrix = shift_operator_users_matrix 
        return shift_operator_users_matrix

    def produce_input_output_pair(self, itemIdxs_input_list, itemIdxs_output_list):
        #Given the itemIdxs (in the original DB) of items of interests it
        #Returns three matrices constituting a single desired input-output pair. The task of the network is to predict 
        #centered ratings in all users given a small collection of centered user ratings. On CENTERED data the missing 
        #data is encoded with zeroes. We build the data set of input output pairs by varying the input/output lists 
        #(which are often, but not necessarily equal)
        
        #The three matrices are:
        #1. Input Filled centered ratings for the given Items, of format len(itemIdxs_input_list) x numUsers
        #2. Output Filled centered ratings for the given Items, of format len(itemIdxs_output_list) x numUsers 
        #3. Locations of entries with actual data for the second matrix, necessary for knowing which entries yield 
        #   actual info, it also has format len(itemIdxs_output_list) x numUsers
        #We allow distinct input and output movies trying to leverage known ratings of similar movies.
        
        nU = self.num_users
        nInput = len(itemIdxs_input_list)
        #Input first:
        X = np.zeros([nInput,nU],dtype="float32")
        for id_u, userIdx in enumerate(self.userIdxs):
            recentered_ratings_u = self.recentered_interactions_dict_by_user[userIdx]
            for id_item, itemIdx in enumerate(itemIdxs_input_list):
                if itemIdx in recentered_ratings_u:
                    X[id_item,id_u] = recentered_ratings_u[itemIdx]
                
        #Output:
        nOutput = len(itemIdxs_output_list)
        Y = np.zeros([nOutput,nU],dtype="float32")
        Z = np.zeros([nOutput,nU], dtype=bool)
        for id_u, userIdx in enumerate(self.userIdxs):
            recentered_ratings_u = self.recentered_interactions_dict_by_user[userIdx]
            for id_item, itemIdx in enumerate(itemIdxs_output_list):
                if itemIdx in recentered_ratings_u:
                    Y[id_item,id_u] = recentered_ratings_u[itemIdx]
                    Z[id_item,id_u] = True

        return X, Y, Z

    def produce_input_output_pair_tensors(self, itemIdxs_input_list, itemIdxs_output_list, fraction_available_data=0.7, num_samples = 5, ):
        X,Y,Z = self.produce_input_output_pair(itemIdxs_input_list, itemIdxs_output_list)     
        As, Bs = self.generate_decompositions(support_matrix = Z,fraction_available_data = fraction_available_data, num_samples=num_samples)
        assert len(As) == len(Bs)
        tensorXs = [torch.tensor(A*X) for A in As]
        tensorYs = [torch.tensor(B*X) for B in Bs]
        tensorBs = [torch.tensor(B) for B in Bs]
        tensorX = torch.cat(tensorXs,0)
        tensorY = torch.cat(tensorYs,0)
        tensorZ = torch.cat(tensorBs)
        return tensorX.unsqueeze(1), tensorY.unsqueeze(1), tensorZ.unsqueeze(1)

    def generate_decompositions(self, support_matrix, fraction_available_data, num_samples):
        #Given a matrix with {0,1} entries Z and a fraction alpha
        #decompose Z = A + B where A has fraction_alpha of the total number of ones.
        #check all entries of Z are integers in {0,1}
        Z = support_matrix
        As = []
        Bs = []
        for k in range(num_samples):
            matrix_size = Z.shape
            A = np.zeros([matrix_size[0],matrix_size[1]], dtype = "float32")
            B = np.zeros([matrix_size[0],matrix_size[1]], dtype = "float32")

            for i in range(matrix_size[0]):
                for j in range(matrix_size[1]):
                    if Z[i,j] == 1:
                        value = np.random.binomial(size = 1, n = 1, p = fraction_available_data)
                        A[i,j] = value[0]

            B = Z-A
            As.append(A)
            Bs.append(B)
        return As, Bs



    def old_produce_input_output_pair_tensors(self, itemIdxs_input_list, itemIdxs_output_list, num_samples = 1):
        #Returns data in the pytorch format of 3-tensors
        #compatible with our operator networks implementation with ONE input feature
        X,Y,Z = self.produce_input_output_pair(itemIdxs_input_list, itemIdxs_output_list)     
        Xs = []
        Ys = []
        Zs = []

        for j in range(num_samples):
            #We compute many support decompositions and produce the data split as samples
            A,B = self.compute_random_support_decomposition(Z,0.3)
            pdb.set_trace()   
            X = torch.Tensor(X).unsqueeze(1)
            Y = torch.Tensor(Y).unsqueeze(1)
            Z = torch.Tensor(Z).unsqueeze(1)
        return X,Y,Z
    

class DataCoreClass:
    def __init__(self, df_items, df_interactions, user_column_name, item_column_name, rating_column_name,rating_lower_bound, rating_upper_bound) -> None:
        self.df_items = df_items
        self.df_interactions = df_interactions
        self.user_column_name = user_column_name
        self.item_column_name = item_column_name
        self.rating_column_name = rating_column_name
        self.rating_lower_bound = rating_lower_bound
        self.rating_upper_bound = rating_upper_bound
    
        df_interactions = self.df_interactions #shortening for nicer pandas code below
        #We extract the vectors of indices of users and items IN THE ORIGINAL DB. We will use these indices throughout        
        userIds_np_array = df_interactions[user_column_name].unique()
        self.userIdxs = userIds_np_array.tolist()#in the original DB, these will not be changed
        itemIds_np_array = df_interactions[item_column_name].unique()
        self.itemIdxs = itemIds_np_array.tolist()#in the original DB, these will not be changed
        #Test claimed ranges hold
        assert (df_interactions[rating_column_name] <= rating_upper_bound).all()
        assert (df_interactions[rating_column_name] >= rating_lower_bound).all()

    def train_test_split_datacores(self, percentage_of_data_for_training):
        #Returns two datacores one for training and one for testing,
        assert percentage_of_data_for_training >= 0.0
        assert percentage_of_data_for_training <= 1.0
        df_interactions = self.df_interactions
        item_column_name = self.item_column_name
        training_row_indices = []
        testing_row_indices = []

        for itemIdx in self.itemIdxs:
            item_info_indices = df_interactions[df_interactions[item_column_name]==itemIdx].index
            item_info_indices = set(item_info_indices.to_list())
            #We split randlomly according to percent
            l = len(item_info_indices)
            training_size = math.ceil(percentage_of_data_for_training*l)
            training_items =  set(random.sample(item_info_indices, training_size))
            testing_items = item_info_indices-training_items
            training_row_indices += list(training_items)
            testing_row_indices += list(testing_items)

        df_interactions_train = df_interactions.loc[training_row_indices]
        df_interactions_test = df_interactions.loc[testing_row_indices]

        #TODO: The overwriting below is awful!! Must be fixed since it can cause errors
        DC_train = DataCoreClass(
            df_items = self.df_items,
            df_interactions= df_interactions_train,
            user_column_name = self.user_column_name, 
            item_column_name = self.item_column_name,
            rating_column_name = self.rating_column_name,
            rating_lower_bound = self.rating_lower_bound,
            rating_upper_bound = self.rating_upper_bound
        )
        DC_train.userIdxs = self.userIdxs
        DC_train.itemIdxs = self.itemIdxs

        DC_test = DataCoreClass(
            df_items = self.df_items,
            df_interactions= df_interactions_test,
            user_column_name = self.user_column_name, 
            item_column_name = self.item_column_name,
            rating_column_name = self.rating_column_name,
            rating_lower_bound = self.rating_lower_bound,
            rating_upper_bound = self.rating_upper_bound
        )
        DC_test.userIdxs = self.userIdxs
        DC_test.itemIdxs = self.itemIdxs

        return DC_train, DC_test



if __name__ == "__main__":
    #We test the class with the movieLens dataset
    import os
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    print(os.chdir(absolute_path)) #Now we are inside the /movie_data_process dir

    df_movies, df_interactions = raw_data_loader(ubound_popular_movies=200, ubound_popular_users=100)
    DC = DataCoreClass(
        df_items = df_movies,
        df_interactions= df_interactions,
        user_column_name = "userId", 
        item_column_name = "movieId",
        rating_column_name = "rating",
        rating_lower_bound = 0.0,
        rating_upper_bound = 5.0
    )
    #Next we split the data
    DC_train, DC_test = DC.train_test_split_datacores(percentage_of_data_for_training = 0.8)

    RS = RecommendationSystem(data_core = DC_train)
    RS.compute_Sigma_and_B_users_matrices()
    RS.compute_shift_operator_users(k_most_correlated = 15)
    shift_operator_1 = RS.shift_operator_users_matrix
    X,Y,Z = RS.produce_input_output_pair(itemIdxs_input_list=[296, 356, 318], itemIdxs_output_list=[296, 356, 318])
    ax = sns.heatmap(X, linewidth=0.5)
    plt.show()
    pdb.set_trace()

    RS2 = RecommendationSystem(data_core = DC_train)
    RS2.compute_Sigma_and_B_users_matrices()
    RS2.compute_shift_operator_users(30)
    shift_operator_2 = RS2.shift_operator_users_matrix

    print(np.linalg.norm(shift_operator_2-shift_operator_1))
    pdb.set_trace()
    ax = sns.heatmap(RS2.B_users_matrix, linewidth=0.5)
    plt.show()
    ax = sns.heatmap(RS.B_users_matrix, linewidth=0.5)
    plt.show()

    print("Done!")
    P_movies_idx = [296, 356, 318, 593, 480, 260, 110, 589, 2571, 527, 1, 457, 150,780, 50,1210 ,592,1196, 2858,32]
    movie_catalog = df_movies[df_movies.movieId.isin(P_movies_idx)]