import pdb
import torch 
import torch.optim as optim
from Collaborative_Filtering_extensions import movieDataExample,recommendClasses
import architectures as archit
import seaborn as sns
import matplotlib.pyplot as plt
import sympy as sp

if __name__ == "__main__":
    #This script trains a recommendation system from scratch on the movieLens dataset
    import os
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    print(os.chdir(absolute_path)) #Now we are guaranteed to be in the /operatornetworks  dir

    # 1. We load the raw data from the database and specify the experimental setup in terms of movies and users
    df_movies, df_interactions = movieDataExample.raw_data_loader(ubound_popular_movies=200, ubound_popular_users=100)
    DC = recommendClasses.DataCoreClass(
        df_items = df_movies,
        df_interactions= df_interactions,
        user_column_name = "userId", 
        item_column_name = "movieId",
        rating_column_name = "rating",
        rating_lower_bound = 0.0,
        rating_upper_bound = 5.0
    )
    #We split the data into disjoint sets for training and testing
    DC_train, DC_test = DC.train_test_split_datacores(percentage_of_data_for_training = 0.7)

    #We call our recommendation systems class and give it access to the training data only,
    RS = recommendClasses.RecommendationSystem(data_core = DC_train)
    RSTest = recommendClasses.RecommendationSystem(data_core = DC_test)

    RS.compute_Sigma_and_B_users_matrices()
    #We define a collaborative filtering shift operator by normalizing rows in the top k most correlated users
    #the idea is that highly correlated users have similar taste so it is a good way to share info between users
    RS.compute_shift_operator_users(k_most_correlated = 20) 
    shift_operator_1 = RS.shift_operator_users_matrix #TODO: This should be a class method
    domain_dim = RS.num_users

    #Next we select a list of movies via their indices, we chose the most popular ones (see movie_catalog for details)
    P_movies_idx = [296, 356, 318, 593, 480, 260, 110, 589, 2571, 527, 1, 457, 150,780, 50,1210 ,592, 1196, 2858,32]
    movie_catalog = df_movies[df_movies.movieId.isin(P_movies_idx)]
    #We compute the training data available for those movies, 
    #the data is placed in the format needed for our operator network classes
    #X,Y,Z = RS.produce_input_output_pair_tensors(itemIdxs_input_list=[296, 356, 318], itemIdxs_output_list=[296, 356, 318])
    X,Y,Z = RS.produce_input_output_pair_tensors(itemIdxs_input_list=P_movies_idx, itemIdxs_output_list=P_movies_idx)
    Xtest, Ytest, Ztest = RSTest.produce_input_output_pair_tensors(itemIdxs_input_list=P_movies_idx, itemIdxs_output_list=P_movies_idx)

    #Exploratory plots on the data before building anything
    figures_path = "./paper_figures/"
    #original data picture
    ax = sns.heatmap(X[:,0,:], cmap="coolwarm", linewidth=0.3)
    figure_name = "Input_data.png"
    plt.savefig(figures_path+figure_name)
    plt.clf()#clears the figure after saving
    #Correlations graph
    ax = sns.heatmap(RS.B_users_matrix, cmap="coolwarm", linewidth=0.0)
    figure_name = "C1_Correlations_Matrix"
    plt.savefig(figures_path+figure_name)
    plt.clf()#clears the figure after saving
    #We plot the shift operator
    ax = sns.heatmap(shift_operator_1, cmap=sns.color_palette("rocket", as_cmap=True), linewidth=0.0)
    figure_name = "C2_shift_operator"
    plt.savefig(figures_path+figure_name)
    plt.clf()#clears the figure after saving

    #EXAMPLE 0: Collaborative filtering
    #We build our X_0 bespoke operator network and evaluate it in the operators we have constructed:
    operator_tuple = (shift_operator_1,)
    x = sp.Symbol("x")
    M = archit.MonomialWordSupport(num_variables=1, allowed_support =[x])#this is the monomial X[0]
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)

    #For the basic collaborative filtering we just multiply the ratings by the normalized correlation matrix
    collaborative_filter_layer = archit.OperatorFilterLayer(num_features_in = 1, num_features_out = 1, monomial_word_support = M)
    torch.nn.init.constant_(collaborative_filter_layer.coefficient_tensor,1.0) 

    #Collaborative_filtering admits only a static analysis:
    YHat = collaborative_filter_layer.forward(X)
    loss = torch.sum((torch.mul(YHat,Ztest)-torch.mul(Ytest,Ztest))**2)/torch.sum(Ztest)#We measure square error only along the entries with data    
    sum = torch.sum(Ztest)
    Y2 = YHat.detach()
    norm2 = torch.norm(torch.mul(Y2,Ztest))
    print(f"norm of vectors with data is {norm2}")
    ax = sns.heatmap(Y2[:,0,:], cmap="coolwarm", linewidth=0.3)
    figure_name = "CF_forecast.png"
    plt.savefig(figures_path+figure_name)
    plt.clf()
    print(f"Out of sample collaborative filtering loss is {loss} on average among {sum} test data points.")

    #Example 1: GNN
    #Next we build our operator network and evaluate it in the operators we have constructed:
    M = archit.MonomialWordSupport(num_variables = 1, allowed_support = 1)
    operator_tuple = (shift_operator_1,) #We start with a single operator (usual GNN)
    M.evaluate_at_operator_tuple(operator_tuple = operator_tuple)

    #Using M we now define our operator network
    #With the monomial support we can build the basic layer,
    #EXAMPLE 1: given partial ratings for a movie produce remaining ratings
    GNN_filter_layer = archit.OperatorFilterLayer(num_features_in = 1, num_features_out = 1, monomial_word_support = M)
 
    #Exploratory analysis of ratings
    ax = sns.heatmap(Y[:,0,:], cmap="coolwarm", linewidth=0.5)
    YHat = GNN_filter_layer.forward(X)
    Y2 = YHat.detach()#this allows turning a tensor parameter to numpy, needed for graphs
    ax = sns.heatmap(Y2[:,0,:], cmap="coolwarm", linewidth=0.5)
    pdb.set_trace()

    #We do a quick training loop
    #We would like to report both in-sample and out-of-sample error


    epsilon = 0.01 #for the learning rate
    optimizer = optim.SGD(filter_layer.parameters(),lr=epsilon)


    counter = 0
    nIters = 200
    while counter < nIters:
        filter_layer.zero_grad()
        YHat = filter_layer.forward(X)
        loss = torch.mean((torch.mul(YHat,Z)-torch.mul(Y,Z))**2)#We measure square error only along the entries with data    
        loss.backward()
        optimizer.step()
        counter+=1
        if counter%5 == 0:
            YHatTest = filter_layer.forward(Xtest)
            out_of_sample_cost = torch.mean((torch.mul(YHatTest,Ztest)-torch.mul(Ytest,Ztest))**2)
            cost = loss.item()
            print(f"Step {counter} Current Loss: {cost}" )
            print(f"Step {counter} Out of sample loss: {out_of_sample_cost}")
            print("\n")



