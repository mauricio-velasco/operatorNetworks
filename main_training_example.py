import torch 
import torch.optim as optim
import architectures as archit
import data as data
import pdb


#1. We specify the experimental parameters. This will take place on 
#a set consisting of num_vertices points with edges given by jumps of fixed size

num_vertices = 293
not_moving_probabilities_vector = [0.05,0.05]
jump_sizes_vector = [1,30]

#tiny example for code verification:
num_vertices = 5
jump_sizes_vector = [1,2]

#Next we construct the training data,
n_samples = 100
x, y = data.dataLab_cycles( num_vertices=num_vertices,
                        not_moving_probabilities_vector=not_moving_probabilities_vector,
                        jump_sizes_vector=jump_sizes_vector,
                        noise_stdev = 0.1,
                        n_samples = n_samples)

ts = data.cycle_operator_tuple( num_vertices=num_vertices,
                    not_moving_probabilities_vector=not_moving_probabilities_vector,
                    jump_sizes_vector=jump_sizes_vector)
#Then we set up the network. For our first experiment it will be a simple linear layer
operator_tuple = ts
M = archit.MonomialWordSupport(num_variables=2, allowed_degree = 3)
M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
#Next we build the basic layer,
filter_layer = archit.OperatorFilterLayer(num_features_in = 1, num_features_out = 1, monomial_word_support = M)
epsilon = 0.001 #for the learning rate
optimizer = optim.SGD(filter_layer.parameters(),lr=epsilon)

#We do a quick training loop
counter = 0
nIters = 3000
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

