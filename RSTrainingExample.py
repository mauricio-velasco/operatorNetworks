import pdb
import torch 
import torch.optim as optim
from Collaborative_Filtering_extensions import movieDataExample,recommendClasses
import architectures as archit
import seaborn as sns
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np




def create_collaborative_filtering_figures_and_tables():
    pass

def make_ratings_plots_on_selected_movies(RS,RSTest,selected_movies_idxs, figures_path, fraction_of_rated_movies_for_image):
    #We compute the training data available for the chosen movies, 
    P_movies_idx = selected_movies_idxs
    Xfull,Yfull,Zfull = RS.produce_input_output_pair(itemIdxs_input_list=P_movies_idx, itemIdxs_output_list=P_movies_idx)
    ax = sns.heatmap(Xfull[:,:], cmap="coolwarm", linewidth=0.3,vmin=-2.5,vmax=2.5)
    figure_name = "Input_data_FULL.png"
    plt.savefig(figures_path+figure_name)
    plt.clf()#clears the figure after saving
    X,Y,Z = RS.produce_input_output_pair_tensors(itemIdxs_input_list=P_movies_idx, itemIdxs_output_list=P_movies_idx,fraction_available_data=fraction_of_rated_movies_for_image, num_samples=1)
    Xtest, Ytest, Ztest = RSTest.produce_input_output_pair_tensors(itemIdxs_input_list=P_movies_idx, itemIdxs_output_list=P_movies_idx)
    #Exploratory plots on the data before building anything
    #partial data picture, used as input for training
    ax = sns.heatmap(X[:,0,:], cmap="coolwarm", linewidth=0.3, vmin=-2.5,vmax=2.5)
    figure_name = "Input_data_PARTIAL.png"
    plt.savefig(figures_path+figure_name)
    plt.clf()#clears the figure after saving

def make_correlations_plot(RS, figure_name):
    #Correlations graph
    ax = sns.heatmap(RS.B_users_matrix, cmap="coolwarm", linewidth=0.0)
    plt.savefig(figures_path+figure_name)
    plt.clf()

def make_nbs_shift_operator_plots(RS, desired_k_most_correlated_nbs_list, figures_path, vmin, vmax):
    operator_list =[]
    for desired_k in desired_k_most_correlated_nbs_list:
        shift_operator_1 = RS.compute_shift_operator_users(k_most_correlated = desired_k)
        ax = sns.heatmap(shift_operator_1, cmap=sns.color_palette("rocket", as_cmap=True), linewidth=0.0,vmin = vmin, vmax=vmax)
        figure_name = "Shift_operator_"+str(desired_k)
        plt.savefig(figures_path+figure_name)
        plt.clf()         
        operator_list.append(shift_operator_1)
    return tuple(operator_list)


def train_model_with_regularization(rs_model, regularization_parameter, training_data_triple, nIters, learning_rate, results_dict, testing_data_triple):
    #given a recommendation system model and a regularization parameter train it using data X,Y,Z dor nIters steps with the given learning_rate
    #It returns the relevant data in a format useful for latter plotting and text construction
    counter = 0
    rs_model_in_sample_loss = []
    rs_model_out_of_sample_loss = []
    optimizer = optim.SGD(rs_model.parameters(),lr = learning_rate)


    while counter < nIters:
        #We compute the in-sample cost and compute gradients (train) accordingly...
        rs_model.zero_grad()
        X = training_data_triple[0]
        Y = training_data_triple[1]
        Z = training_data_triple[2]
        YHat = rs_model.forward(X)
        vparms = torch.nn.utils.parameters_to_vector(rs_model.parameters())
        loss = torch.sum(((YHat*Z)-(Y*Z))**2)/torch.sum(Z)+regularization_parameter*(torch.norm(vparms)**2)    
        cost = loss.item()
        rs_model_in_sample_loss.append(cost)
        loss.backward()
        optimizer.step()
        counter+=1

        if testing_data_triple:
            #If testing data is provided then the algorithm computes the out of sample loss in each step of the training
            assert len(testing_data_triple) == 3, "Testing data must consist of three tensors X,Y,Z"
            Xtest = testing_data_triple[0]
            Ytest = testing_data_triple[1]
            Ztest = testing_data_triple[2]

            YHatTest = rs_model.forward(Xtest)
            out_of_sample_cost = torch.sum(((YHatTest*Ztest)-(Ytest*Ztest))**2)/torch.sum(Ztest)
            oso_cost = out_of_sample_cost.item()
            rs_model_out_of_sample_loss.append(oso_cost)

        #Finally we add the results to the results dictionary...
        results_dict["rs_model_in_sample_loss"] = rs_model_in_sample_loss
        results_dict["nIters"] = nIters
        results_dict["min_rs_model_in_sample_loss"] = min(rs_model_in_sample_loss)

        if testing_data_triple:
            results_dict["rs_model_out_of_sample_loss"] = rs_model_out_of_sample_loss
            results_dict["min_rs_model_out_of_sample_loss"] = min(rs_model_out_of_sample_loss)


def make_and_save_individual_plots(results_array,figures_path):
    assert all(["model_name" in row_dict for row_dict in results_array]),"Every row_dict from results must contain a model_name"
    for row_dict in results_array:
        figure_name = row_dict["model_name"]
        In_sample_losses = row_dict["rs_model_in_sample_loss"] 
        fig, ax = plt.subplots()
        nIters = row_dict["nIters"]
        Xs = range(nIters)
        ax.set(xlabel='Iteration', ylabel='MSE')
        ax.grid()
        ax.axis(xmin = 0,xmax = nIters)
        ax.plot(Xs,In_sample_losses, label="Is_losses") #Hace la grafica, poniendo puntos en las parebas correspondientes 
        if "rs_model_out_of_sample_loss" in row_dict:        
            Outof_sample_losses = row_dict["rs_model_out_of_sample_loss"]
            ax.plot(Xs,Outof_sample_losses, label="Os_losses") #Hace la grafica, poniendo puntos en las parebas correspondientes            
        ax.legend()
        fig.savefig(figures_path+figure_name+".png") #Graba un archivo con la imagen
        fig.clf()


def make_and_save_out_of_sample_plots(results_array,figures_path):
    assert all(["model_name" in row_dict for row_dict in results_array]),"Every row_dict from results must contain a model_name"
    assert all(["rs_model_out_of_sample_loss" in row_dict for row_dict in results_array]),"Every row_dict must_have out_of_sample_data"

    fig, ax = plt.subplots()
    figure_name = "Out_of_sample_plots"
    first = True

    for row_dict in results_array:
        if first:
            first=False
            nIters = row_dict["nIters"]
            Xs = range(nIters)
            ax.axis(xmin = 0,xmax = nIters)
            ax.set(xlabel='Iteration', ylabel='MSE')


        legend_name = row_dict["model_name"] + "_OOs"
        Outof_sample_losses = row_dict["rs_model_out_of_sample_loss"]
        ax.plot(Xs,Outof_sample_losses, label=legend_name) #Hace la grafica, poniendo puntos en las parebas correspondientes            

    ax.grid()    
    ax.legend()
    fig.savefig(figures_path+figure_name+".png") #Graba un archivo con la imagen
    fig.clf()



if __name__ == "__main__":
    #This script trains several recommendation system from scratch on the movieLens dataset
    #and allows reproduction of figures and data tables as appearing in the article.
    import os
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    print(os.chdir(absolute_path)) #Now we are guaranteed to be in the /operatornetworks  dir
    # 1. We load the raw data from the database and specify the experimental setup in terms of movies and users
    df_movies, df_interactions = movieDataExample.raw_data_loader(ubound_popular_movies=200, ubound_popular_users=300)
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
    DC_train, DC_test = DC.train_test_split_datacores(percentage_of_data_for_training = 0.8)

    #We call our recommendation systems class and give it access to the training data only, 
    #and testing data only respectively
    RS = recommendClasses.RecommendationSystem(data_core = DC_train)
    RSTest = recommendClasses.RecommendationSystem(data_core = DC_test)
    #Next we compute the correlation matrix on all available data
    RS.compute_Sigma_and_B_users_matrices()
    #We establish the directory where figures are built,
    figures_path = "./paper_figures/"


    #Next we select a list of movies via their indices, choosing the most popular ones (see movie_catalog for details)
    P_movies_idx = [296, 356, 318, 593, 480, 260, 110, 589, 2571, 527, 1, 457, 150,780, 50,1210 ,592, 1196, 2858,32]
    movie_catalog = df_movies[df_movies.movieId.isin(P_movies_idx)]
    #and we do some exploratory data analysis images on the data coming from those movies
    make_ratings_plots_on_selected_movies(RS=RS,
                                          RSTest=RSTest,
                                          selected_movies_idxs=P_movies_idx,
                                          figures_path=figures_path,
                                          fraction_of_rated_movies_for_image=0.6
                                          )



    #Next we look at the shift operators obtained from collaborative filtering:
    make_correlations_plot(RS = RS, figure_name= "C1_Correlations_matrix")
    full_operator_tuple = make_nbs_shift_operator_plots(RS=RS, desired_k_most_correlated_nbs_list=[10,15,20],figures_path=figures_path,vmin = 0.0,vmax = 1.0)


    #*********E X A M P L E S ************************************************************************
    #EXAMPLE 0: Collaborative filtering
    #We build our X_0 bespoke operator network and evaluate it in the operators we have constructed:
    operator_tuple = (full_operator_tuple[0],)
    x = sp.Symbol("x")
    M = archit.MonomialWordSupport(num_variables=1, allowed_support =[x])#this is the monomial X[0]
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    #For the basic collaborative filtering we just multiply the ratings by the normalized correlation matrix
    collaborative_filter_layer = archit.OperatorFilterLayer(num_features_in = 1, num_features_out = 1, monomial_word_support = M)
    torch.nn.init.constant_(collaborative_filter_layer.coefficient_tensor, 1.0)

    #The behavior of the resulting operator network is measured by reporting the mean squared error in the TESTING SET 
    Xtest, Ytest, Ztest = RSTest.produce_input_output_pair_tensors(
        itemIdxs_input_list=P_movies_idx, 
        itemIdxs_output_list=P_movies_idx,
        fraction_available_data= 0.8,
        num_samples = 1)

    #Collaborative_filtering admits only a static analysis:
    YHat = collaborative_filter_layer.forward(Xtest)
    loss = torch.sum(((YHat*Ztest)-(Ytest*Ztest))**2)/torch.sum(Ztest)#We measure square error only along the entries with data    
    Y2 = YHat.detach()
    ax = sns.heatmap(Y2[:,0,:], cmap="coolwarm", linewidth=0.3, vmin=-2.5, vmax =2.5)
    figure_name = "CF_forecast.png"
    plt.savefig(figures_path+figure_name)
    plt.clf()
    closs = loss

    results_array = [] #We will put the results in an array of dicts

    #Example 1: GNN HERE:
    #Next we build our operator network and evaluate it in the operators we have constructed, allowing a slightly higher degree
    M = archit.MonomialWordSupport(num_variables = 1, allowed_support = 6)#we SET SUPPORT
    #operator_tuple = (full_operator_tuple[0],) #The full operator tuple was defined above
    operator_tuple = (full_operator_tuple[2],) #The full operator tuple was defined above, the latter diffuses more slowly so it is harder to beat
    M.evaluate_at_operator_tuple(operator_tuple = operator_tuple)
    #Using M we now define our operator network
    #With the monomial support we can build the basic layer,
    #EXAMPLE 1: given partial ratings for a movie produce remaining ratings
    #parameters of the model
    num_features_in = 1
    num_features_out = 1
    monomial_word_support = M
    #training and testing data
    X, Y, Z = RS.produce_input_output_pair_tensors(
        itemIdxs_input_list=P_movies_idx, 
        itemIdxs_output_list=P_movies_idx,
        fraction_available_data= 0.8,
        num_samples = 2)

    Xtest, Ytest, Ztest = RSTest.produce_input_output_pair_tensors(
        itemIdxs_input_list=P_movies_idx, 
        itemIdxs_output_list=P_movies_idx,
        fraction_available_data= 0.8,
        num_samples = 2)


    nIters = 1500 #we use the same number of iterations for all models
    #Now onto the actual computation...
    result_row = dict([("model_name","GNN_reg_0.0")])
    GNN_filter_layer = archit.OperatorFilterLayer(num_features_in = num_features_in, num_features_out = num_features_out, monomial_word_support = monomial_word_support)
    train_model_with_regularization(
        rs_model = GNN_filter_layer,
        regularization_parameter = 0.0,
        training_data_triple = [X,Y,Z],
        nIters = nIters,
        testing_data_triple = [Xtest,Ytest,Ztest],
        results_dict=result_row,
        learning_rate=0.005)
    results_array.append(result_row)


    result_row = dict([("model_name","GNN_reg_0.12")])
    GNN_filter_layer = archit.OperatorFilterLayer(num_features_in = num_features_in, num_features_out = num_features_out, monomial_word_support = monomial_word_support)
    train_model_with_regularization(
        rs_model = GNN_filter_layer,
        regularization_parameter = 0.12,
        training_data_triple = [X,Y,Z],
        nIters = nIters,
        testing_data_triple = [Xtest,Ytest,Ztest],
        results_dict=result_row,
        learning_rate=0.005)
    results_array.append(result_row)


    result_row = dict([("model_name","GNN_reg_0.5")])
    GNN_filter_layer = archit.OperatorFilterLayer(num_features_in = num_features_in, num_features_out = num_features_out, monomial_word_support = monomial_word_support)
    train_model_with_regularization(
        rs_model = GNN_filter_layer,
        regularization_parameter = 0.5,
        training_data_triple= [X,Y,Z],
        nIters = nIters,
        testing_data_triple= [Xtest,Ytest,Ztest],
        results_dict=result_row,
        learning_rate=0.005)
    results_array.append(result_row)



    #Example 2: Two operators network
    #Next we build our operator network and evaluate it in the operators we have constructed,
    M = archit.MonomialWordSupport(num_variables = 2, allowed_support = 2)#we SET SUPPORT
    operator_tuple = (full_operator_tuple[0],full_operator_tuple[2]) #The full operator tuple was defined above
    #operator_tuple = (full_operator_tuple[0],full_operator_tuple[1],full_operator_tuple[2])
    M.evaluate_at_operator_tuple(operator_tuple = operator_tuple)
    #Using M we now define our operator network
    #With the monomial support we can build the basic layer,
    num_features_in = 1
    num_features_out = 1
    monomial_word_support = M
    
    #nIters = nIters #we use the same number of iterations for all 2ONN models
    result_row = dict([("model_name","2ONN_reg_0.0")])
    ONN_filter_layer = archit.OperatorFilterLayer(num_features_in=num_features_in, num_features_out=num_features_out,monomial_word_support=monomial_word_support)
    train_model_with_regularization(
        rs_model = ONN_filter_layer,
        regularization_parameter = 0.0,
        training_data_triple = [X,Y,Z],
        nIters = nIters,
        testing_data_triple = [Xtest,Ytest,Ztest],
        results_dict=result_row,
        learning_rate=0.005)

    results_array.append(result_row)

    result_row = dict([("model_name","2ONN_reg_0.125")])
    ONN_filter_layer = archit.OperatorFilterLayer(num_features_in=num_features_in, num_features_out=num_features_out,monomial_word_support=monomial_word_support)
    train_model_with_regularization(
        rs_model = ONN_filter_layer,
        regularization_parameter = 0.125,
        training_data_triple = [X,Y,Z],
        nIters = nIters,
        testing_data_triple = [Xtest,Ytest,Ztest],
        results_dict=result_row,
        learning_rate=0.005)

    results_array.append(result_row)


    #Finally we create plots of the data we computed...
    figures_path = "./paper_figures/"
    make_and_save_individual_plots(results_array,figures_path)
    make_and_save_out_of_sample_plots(results_array,figures_path)
    #Figures completed...

 
  