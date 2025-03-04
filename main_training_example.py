import pdb
import torch 
import torch.optim as optim
import architectures as archit
import data as data


#1. We specify the experimental parameters. This will take place on 
#a set consisting of num_vertices points with edges given by a cycle obtained by additive jumps 
#of fixed size on the residue classes 0,...,292 mod 293.

num_vertices = 293
not_moving_probabilities_vector = [0.05,0.05]
jump_sizes_vector = [1,30]

#tiny example for code verification:
#num_vertices = 5
#jump_sizes_vector = [1,2]

#Next we construct the training data,
n_samples = 100
x, y = data.dataLab_cycles( num_vertices=num_vertices,
                        not_moving_probabilities_vector=not_moving_probabilities_vector,
                        jump_sizes_vector=jump_sizes_vector,
                        noise_stdev = 0.1,
                        n_samples = n_samples)
#The x and y in the training data are of shape [100, 1, Num_vertices] because it consists of a 

ts = data.cycle_operator_tuple( num_vertices=num_vertices,
                    not_moving_probabilities_vector=not_moving_probabilities_vector,
                    jump_sizes_vector=jump_sizes_vector)
#Having our operators we will define the operator network
#For that we first need a monomial support object which evaluates noncommutative polyomials
#In our fixed operators
operator_tuple = ts #The operators are a tuple of tensors of size num_vertices x num_vertices
M = archit.MonomialWordSupport(num_variables=2, allowed_support = 3)
M.evaluate_at_operator_tuple(operator_tuple = operator_tuple)

#With the monomial support we can build the basic layer,
filter_layer = archit.OperatorFilterLayer(num_features_in = 1, num_features_out = 1, monomial_word_support = M)
filter_layer = archit.NeuralReLuOperatorFilterLayer(num_features_in = 1, num_features_out = 1, monomial_word_support = M)
epsilon = 0.001 #for the learning rate
optimizer = optim.SGD(filter_layer.parameters(),lr=epsilon)

#We do a quick training loop
counter = 0
nIters = 2000
while counter < nIters:
    filter_layer.zero_grad()
    yHat = filter_layer.forward(x)
    loss = torch.mean((yHat-y)**2)
    loss.backward()
    optimizer.step()
    counter+=1
    if counter%50 == 0:
        cost = loss.item()
        print(f"Step {counter} Current Loss: {cost}\n" )

