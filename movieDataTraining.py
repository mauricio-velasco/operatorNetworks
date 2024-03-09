import torch
import torch.nn as nn 
import torch.optim as optim
import architectures as archit
import movieDataLoader as moviedata
import data as data
import pdb

class simpleMovieRecommendationNet(nn.Module):
    def __init__(self, monomial_word_support, stw_index):        
        super().__init__()
        self.stw_index = stw_index
        neural_filter_layer = archit.NeuralReLuOperatorFilterLayer(num_features_in = 1, 
                                                                   num_features_out = 64, 
                                                                   monomial_word_support = monomial_word_support)
        self.nfl = neural_filter_layer
    def forward(self,x):
        y = self.nfl(x)
        m = y.mean(dim = 1) #Check whether mean is occuring in the right dimension
        index = self.stw_index #Star_wars index
        return m[:,index] #Check whether mean is occuring in the right dimension


#0. We specify the experimental parameters.
num_vertices = 500 #This example uses the 500
allowed_degree = 3

#1. We build the data training pairs

#Next we construct the training data,
n_samples = 100 #Fix the number of samples we want to extract,
""" The structure of the training samples is the following:
(1) x[i,j,k] is a 3-tensor such that x[i,:,:] is a matrix (actually a row_vector) 
of format  1 x num_vertices containing the score of all num_vertices movies for 
person i (more precisely x[i,0,j] = rating of movie j for person i)
(2) y[i,j] is a matrix (more precisely a column vector) where y[i,:]=y[i,0] is a 
scalar, the rating given for the STARWARS movie by person i.
"""
#xtrain, ytrain = moviedata.get_batch( num_vertices=num_vertices,n_samples = n_samples)
#TODO: plan validation split and cross-validation 

#Overwrite data construction for downstream testing
#TODO:Remove this when the MovieData Loader is ready
num_vertices = 7
not_moving_probabilities_vector = [0.05,0.05]
jump_sizes_vector = [1,2]
xtrain, ytrain = data.dataLab_cycles( num_vertices=num_vertices,
                        not_moving_probabilities_vector=not_moving_probabilities_vector,
                        jump_sizes_vector=jump_sizes_vector,
                        noise_stdev = 0.1,
                        n_samples = n_samples)
ytrain = ytrain[:,:,4] #simulating STARWARS index choice

#2. We build the operators
#assert num_vertices in [500,1000] #Allowed range for the vertices
#ts = moviedata.build_operator_tuple( num_vertices=num_vertices) #returns a pair of operators

#For now overwrite operator definition for downstream testing:
#TODO:Remove this when the MovieData Loader is ready
ts = data.cycle_operator_tuple( num_vertices=num_vertices,
                    not_moving_probabilities_vector=not_moving_probabilities_vector,
                    jump_sizes_vector=jump_sizes_vector)
operator_tuple = ts

#3. We build the noncommuting products of our operators using the MonomialWordSupport object
M = archit.MonomialWordSupport(num_variables = 2, allowed_degree = allowed_degree)
M.evaluate_at_operator_tuple(operator_tuple = operator_tuple)

#4. With the monomial support we construct the simple network
net = simpleMovieRecommendationNet(monomial_word_support= M,stw_index =[4])

#5. Finally, we train the network:
epsilon = 0.01 #value for the learning rate #TODO: needs to be cross-validated
optimizer = optim.SGD(net.parameters(),lr = epsilon)

counter = 0
nIters = 5000
while counter < nIters:
    net.zero_grad()
    yHat = net.forward(xtrain)
    loss = torch.mean((yHat-ytrain)**2)
    loss.backward()
    optimizer.step()
    counter+=1
    if counter%50 == 0:
        cost = loss.item()
        print(f"Step {counter} Current Loss: {cost}\n" )
