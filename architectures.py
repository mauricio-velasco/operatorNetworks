import sympy as sp
import torch
import torch.nn as nn
import pdb

class MonomialWordSupport:
    #This class specifies the number of noncommuting variables and the support of 
    #all the multivariate polynomials we wish to consider.
    #It is capable to evaluate such monomials on a given operator tuple (it then becomes evaluated)
    #Once evaluated it can be applied to a vector f (returning the vector of evaluated monomial words applied to f )
    def __init__(self, num_variables, allowed_degree) -> None:
        self.num_variables = num_variables
        self.allowed_degree = allowed_degree
        k = self.num_variables
        X = sp.symbols('X0:'+str(k), commutative=False)#We will always use the variables X_0,...X_{k-1}
        self.variables = X
        #We define the allowed monomial_words
        #TODO: In a more general framework the monomial words can be given as input by the user 
        #(because there is no reason to restrict only by degree)
        self.monomial_words = list(sp.itermonomials(self.variables, self.allowed_degree))
        self.is_evaluated = False

    def evaluate_at_operator_tuple(self, operator_tuple): 
        #Given an operator tuple computes the self.operator_evaluated_monomial_words
        #and sets self.is_evaluated to True and the self.current_operator_tuple

        #First, we check validity of input
        assert len(operator_tuple) == self.num_variables, "Operator tuple size must match the number of variables in the support object."   
        s = operator_tuple[0].shape
        assert len(s)==2 
        assert s[0]==s[1], "Operators must be square matrices"
        for operator in operator_tuple:
            assert s == operator.shape, "All operators must be square matrices and act in the same space"
        #Carry out evaluation of monomial words...
        self.operator_evaluated_monomial_words = []
        X = self.variables
        for word in self.monomial_words:
            parts = word.args
            word_operator = torch.eye(s[0])#We initialize each monomial word with an identity
            for part in reversed(parts):
                if part.func == X[0].func:
                    op_index = X.index(part)
                    word_operator = operator_tuple[op_index] @ word_operator
                if part.func == (X[0]**2).func:                    
                    var_symbol = part.args[0]
                    op_index = X.index(var_symbol)
                    op = operator_tuple[op_index]
                    var_power = int(part.args[1])
                    word_operator = torch.matrix_power(op,var_power) @ word_operator

            self.operator_evaluated_monomial_words.append(word_operator)             
        #Evaluation has been achieved, so we set our object to the evaluated mode.
        self.current_operator_tuple = operator_tuple
        self.is_evaluated = True
        self.operator_domain_dim = s[0]


    def monomial_words_forward(self,x):
        #Given a matrix f which we think as a collection of rows we will produce an evaluation tensor
        #E(a1,a2,alpha) = Component a_2 of the vector of x^{\alpha}(T)(f_{a_1})
        result = []
        for row_vector in x:
            TN = torch.stack([torch.mv(op, row_vector) for op in self.operator_evaluated_monomial_words ])
            result.append(torch.transpose(TN,1,0))
        return torch.stack(result)

    def num_monomial_words(self):
        return len(self.monomial_words)

    def operator_matrix_from_coeffs(self, coefficients_vector):
        total_length = self.num_monomial_words()
        res = sum([coefficients_vector[k] * self.operator_evaluated_monomial_words[k] for k in range(total_length)])        
        return res

class OperatorFilterLayer(nn.Module):
    def __init__(self, num_features_in,num_features_out, monomial_word_support):        
        super().__init__()
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.monomial_word_support = monomial_word_support
        num_coeffs = monomial_word_support.num_monomial_words()
        #the coefficient tensor remembers the coefficients of all the involved polynomials, is an B x A x num_monomials tensor
        coefficient_tensor = torch.Tensor(self.num_features_out, self.num_features_in, num_coeffs)
        self.coefficient_tensor = nn.Parameter(coefficient_tensor) #The tensor of coefficients is the trainable parameter
        assert monomial_word_support.is_evaluated, "The monomial support must be evaluated in an operator tuple to define and train a network."
        #Initialization: TODO: think of a good initialization. The coefficients should sum to one along the third direction.
        nn.init.uniform_(self.coefficient_tensor)

    def forward(self,x):
        #By torch conventions matrices are collections of ROWS so we will think of x this way,
        data_shape = x.shape 
        #pdb.set_trace()       
        assert self.monomial_word_support.operator_domain_dim == data_shape[1] and self.num_features_in == data_shape[0], "Evaluation point x must be matrix of size domain_dim x num_features_in"
        M = self.monomial_word_support
        evaluations_tensor = M.monomial_words_forward(x)        
        coefficients_tensor = self.coefficient_tensor
        #The following contraction defines the filter...
        res = torch.tensordot( coefficients_tensor, evaluations_tensor,dims = ([1,2],[0,2]))
        return res




#TESTS:
global tol 
tol = 1e-5


def monomial_evaluation_test():
    t0 = torch.rand(2,2)
    t1 = torch.rand(2,2)
    operator_tuple = (t0,t1)
    M = MonomialWordSupport(num_variables=2, allowed_degree = 4)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    X = M.variables
    new_monomial = X[0]*X[0]*X[1]*X[1]    
    mon_index = M.monomial_words.index(new_monomial)
    N2 = t0 @ t0 @ t1 @ t1 - M.operator_evaluated_monomial_words[mon_index]
    assert torch.norm(N2) < tol, "ERROR in evaluation"

def operator_matrix_from_coeffs_test():
    #set up the support, 2 variables degree at most three
    M = MonomialWordSupport(num_variables=2, allowed_degree = 3)
    #We define an operator tuple to evaluate the monomial support as follows,
    arr1 = [[0.5, 0], [0, 0.5]]
    arr2 = [[0, 0.5], [0.5, 0.0]]
    #arr2 = [[0, 0.5, 3],[0, 0.5, 3] ]
    t1 = torch.Tensor(arr1)
    t2 = torch.Tensor(arr2)
    operator_tuple = (t2,t1)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)#Evaluation of the monomial support at the given op tuple.
    #Next we compute the operator in two ways and compare the results
    total_monomial_length = M.num_monomial_words()
    coefficients_vector = torch.zeros(total_monomial_length)
    monomial_index = 5
    coefficients_vector[monomial_index] = 3.0 
    matrix_res = M.operator_matrix_from_coeffs(coefficients_vector=coefficients_vector)
    computed_res = 3.0*M.operator_evaluated_monomial_words[monomial_index]
    normres = torch.norm(matrix_res-computed_res)
    assert normres < tol

def forward_layer_test():
    #set up the support, 2 variables degree at most three
    M = MonomialWordSupport(num_variables=2, allowed_degree = 3)
    #We define an operator tuple to evaluate the monomial support as follows,
    arr1 = [[0.5, 0], [0, 0.5]]
    arr2 = [[0, 0.5], [0.5, 0.0]]
    #arr2 = [[0, 0.5, 3],[0, 0.5, 3] ]
    t1 = torch.Tensor(arr1)
    t2 = torch.Tensor(arr2)
    operator_tuple = (t2,t1)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)#Evaluation of the monomial support at the given op tuple.
       
    #With the evaluated support, we build a filter layer with feature sizes 2,3
    filter_layer = OperatorFilterLayer(num_features_in = 2, num_features_out = 3, monomial_word_support = M)
    coefficients_tensor = filter_layer.coefficient_tensor
    H10 = M.operator_matrix_from_coeffs(coefficients_vector = coefficients_tensor[1,0,:])
    H11 = M.operator_matrix_from_coeffs(coefficients_vector = coefficients_tensor[1,1,:])
    #We create a random tensor to test the output 
    x_tensor = torch.Tensor(2,2) 
    nn.init.uniform_(x_tensor) #the two 2-diml features are the rows of x_tensor
    #print(x_tensor)
    #We compute the output according to the layer
    y_tensor = filter_layer.forward(x_tensor)
    target_row = y_tensor[1,:]
    #and according to the block formula
    computed_row = torch.mv(H10, x_tensor[0,:]) + torch.mv(H11, x_tensor[1,:]) 
    diff = target_row-computed_row
    assert torch.norm(diff) < tol, "ERROR, formula fails to match to desireable accuracy"






if __name__ == "__main__":
    monomial_evaluation_test() #Verifies that the evaluation behaves correctly. TODO: Should be made into a unit test
    operator_matrix_from_coeffs_test() #Verifies that the evaluation of operators behaves correctly. TODO: Should be made into a unit test
    forward_layer_test() #Verifies that the contraction in the forward layer agrees with the simple block-formula

    #One defines an operator tuple as follows
    arr1 = [[0.5, 0], [0, 0.5]]
    arr2 = [[0, 0.5], [0.5, 0.0]]
    #arr2 = [[0, 0.5, 3],[0, 0.5, 3] ]
    t1 = torch.Tensor(arr1)
    t2 = torch.Tensor(arr2)
    operator_tuple = (t2,t1)
    M = MonomialWordSupport(num_variables=2, allowed_degree = 3)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    x = [[1.0,1.0], [3.0,1.0], [5.0,2.0]] #We will apply the function to the rows of a matrix. 
    x_tensor = torch.Tensor(x)#Flip coordinates
    EvT = M.monomial_words_forward(x_tensor)
    #Next we build layers,
    filter_layer = OperatorFilterLayer(num_features_in = 3, num_features_out = 4, monomial_word_support = M)
    #Our layer has two input and two output dimensions so
    x = [[1.0,1.0], [3.0,1.0],[4.0,1.0]] #We will apply the function to the rows of a matrix. 
    x_tensor = torch.Tensor(x) #We think the input is a matrix and we want to evaluate the operator in the
    res_tensor = filter_layer.forward(x_tensor)



    #All evaluations at done at the level of the support so it is a reasonable place to evaluate performance.
    dim = 100 #number of nodes of the graph, 5000 takes a few mins
    t0 = torch.rand(dim,dim)
    t1 = torch.rand(dim,dim)
    operator_tuple = (t0,t1)
    M = MonomialWordSupport(num_variables=2, allowed_degree = 4)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    #Next we build layers,
    filter_layer = OperatorFilterLayer(num_features_in = 2, num_features_out = 2, monomial_word_support = M)

