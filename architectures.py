import sympy as sp
import torch
import torch.nn as nn

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


    def monomial_words_forward(self,vector_x):
        result = [torch.mv(op, vector_x) for op in self.operator_evaluated_monomial_words ]
        return result        



    def num_monomial_words(self):
        return len(self.monomial_words)
        


class OperatorFilterLayer(nn.Module):
    def __init__(self, num_features_in,num_features_out, monomial_word_support):        
        super().__init__()
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.monomial_word_support = monomial_word_support
        num_coeffs = monomial_word_support.num_monomial_words()
        coefficient_tensor = torch.Tensor(self.num_features_out, self.num_features_in, num_coeffs)
        self.coefficient_tensor = nn.Parameter(coefficient_tensor) #The tensor of coefficients is the trainable parameter
        assert monomial_word_support.is_evaluated, "The monomial support must be evaluated in an operator tuple to define and train a network."
        #Initialization: TODO: think of a good initialization. The coefficients should sum to one along the third direction.
        nn.init.uniform_(self.coefficient_tensor)

    def forward(self,x):
        data_shape = x.shape        
        assert self.operator_domain_dim == data_shape[0] and self.num_features_in == data_shape[1], "Evaluation point x must be matrix of size domain_dim x num_features_in"
        for b in range(self.num_features_out):
            pass







def operator_test():
    t0 = torch.rand(2,2)
    t1 = torch.rand(2,2)
    operator_tuple = (t0,t1)
    M = MonomialWordSupport(num_variables=2, allowed_degree = 4)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    X = M.variables
    new_monomial = X[0]*X[0]*X[1]*X[1]    
    mon_index = M.monomial_words.index(new_monomial)
    N2 = t0 @ t0 @ t1 @ t1 - M.operator_evaluated_monomial_words[mon_index]
    assert torch.norm(N2) < 1e-5, "ERROR in evaluation"

if __name__ == "__main__":
    operator_test() #Verifies that the evaluation behaves correctly. TODO: Should be made into a unit test
    #One defines an operator tuple as follows
    arr1 = [[0.5, 0], [0, 0.5]]
    arr2 = [[0, 0.5], [0.5, 0.0]]
    #arr2 = [[0, 0.5, 3],[0, 0.5, 3] ]
    t1 = torch.Tensor(arr1)
    t2 = torch.Tensor(arr2)
    operator_tuple = (t2,t1)
    M = MonomialWordSupport(num_variables=2, allowed_degree = 3)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    vector_x = [1.0,1.0]
    x_tensor = torch.Tensor(vector_x)#Flip coordinates
    M.monomial_words_forward(x_tensor)

    #All evaluations at done at the level of the support so it is a reasonable place to evaluate performance.
    dim = 100 #number of nodes of the graph, 5000 takes a few mins
    t0 = torch.rand(dim,dim)
    t1 = torch.rand(dim,dim)
    operator_tuple = (t0,t1)
    M = MonomialWordSupport(num_variables=2, allowed_degree = 4)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)

    #Next we build layers,
    filter_layer = OperatorFilterLayer(num_features_in = 2, num_features_out = 2, monomial_word_support = M)

