import sympy as sp
import torch
import torch.nn as nn
import pdb

class MonomialWordSupport:
    #This class specifies the number of noncommuting variables and the support of 
    #all the multivariate polynomials we wish to consider.
    #It is capable to evaluate such monomials on a given operator tuple (it then becomes evaluated)
    #Once evaluated it can be applied to a vector f (returning the vector of evaluated monomial words applied to f )
    def __init__(self, num_variables, allowed_support) -> None:
        assert isinstance(num_variables,int)
        self.num_variables = num_variables
        self.is_evaluated = False
        #Next we define the set of allowed monomials according to the input structure
        if isinstance(allowed_support,int):
            assert allowed_support >=1, "Bound in degree of polynomials must be at least one"
            #Implicitly means that the support is the set of all monomials of degree at most allowed_support
            self.allowed_degree = allowed_support
            X = sp.symbols('X0:'+str(self.num_variables), commutative=False)#We will always use the variables X_0,...X_{k-1}
            self.variables = X
            #We define the allowed monomial_words
            self.monomial_words = list(sp.itermonomials(self.variables, self.allowed_degree))
        else:
            #The allowed support in this case is a list of sympy expressions...
            #TODO: Add Check this is indeed the case...
            total_expression = sum(allowed_support)
            ambient_variables = total_expression.atoms(sp.Symbol) #searches for basic symbols in an expression to obtain indices
            ambient_variables = list(ambient_variables)
            self.variables = ambient_variables            
            assert len(self.variables)<= self.num_variables, "Defining expressions must involve at most num_variables variables"
            self.monomial_words = allowed_support                

    
    def evaluate_expression_in_operator_tuple(self,word_expression, ambient_variables):
        #This function evaluates an expression on the ambient variables in our chosen operator tuple
        #It uses the tree structure for sympy expressions as documented here
        #https://docs.sympy.org/latest/tutorials/intro-tutorial/manipulation.html

        assert hasattr(self, "operator_domain_dim"), "The monomial support must be evaluated in an operator tuple before evaluating expressions"
        word_operator = torch.eye(self.operator_domain_dim)#We initialize each monomial word with an identity
        if word_expression.func == sp.Mul:
            for subexpr in word_expression.args:
                new_part = self.evaluate_expression_in_operator_tuple(word_expression=subexpr, ambient_variables=ambient_variables)
                word_operator = word_operator @ new_part 

        elif word_expression.func == sp.Add:
            word_operator = torch.zeros([self.operator_domain_dim,self.operator_domain_dim])
            for subexpr in word_expression.args:
                new_part = self.evaluate_expression_in_operator_tuple(word_expression=subexpr, ambient_variables=ambient_variables)
                word_operator = word_operator + new_part 

        elif word_expression.func == sp.Pow:
            arg_list = word_expression.args
            subexpr = arg_list[0]
            previous_step_operator = self.evaluate_expression_in_operator_tuple(word_expression = subexpr, ambient_variables=ambient_variables)
            var_power = int(arg_list[1])
            word_operator = torch.matrix_power(previous_step_operator,var_power)

        elif word_expression.func == sp.Symbol:
            index = ambient_variables.index(word_expression)
            word_operator = self.current_operator_tuple[index]

        elif word_expression.func == sp.core.numbers.One:
            word_operator = torch.eye(self.operator_domain_dim)

        else:
            raise AssertionError("Unhandled expression types in input")

        return word_operator

    def compute_word_operators_from_monomials(self, support_as_simpy_expressions_list):
        #Given a collection of simpy expressions this evaluates them
        #the ambient variables are specified by the class constructor
        ambient_variables = self.variables
        operator_evaluated_monomial_words = []
        for word_expression in support_as_simpy_expressions_list:
            word_operator = self.evaluate_expression_in_operator_tuple(word_expression=word_expression,ambient_variables=ambient_variables)
            operator_evaluated_monomial_words.append(word_operator)
        return operator_evaluated_monomial_words


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
        self.current_operator_tuple = operator_tuple
        self.operator_domain_dim = s[0]
        #Finally carry out evaluation of monomial words in their respective operators...
        self.operator_evaluated_monomial_words = self.compute_word_operators_from_monomials(self.monomial_words)
        self.is_evaluated = True




    def old_evaluate_at_operator_tuple(self, operator_tuple): 
        #Given an operator tuple computes the self.operator_evaluated_monomial_words
        #and sets self.is_evaluated to True and the self.current_operator_tuple

        #First, we check validity of input
        assert len(operator_tuple) == self.num_variables, "Operator tuple size must match the number of variables in the support object."   
        s = operator_tuple[0].shape
        assert len(s)==2 
        assert s[0]==s[1], "Operators must be square matrices"
        for operator in operator_tuple:
            assert s == operator.shape, "All operators must be square matrices and act in the same space"
        self.current_operator_tuple = operator_tuple
        self.operator_domain_dim = s[0]

        #Carry out evaluation of monomial words...
        self.operator_evaluated_monomial_words = []
        X = self.variables
        for word in self.monomial_words:
            #TODO:Fix this!!
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

        self.is_evaluated = True


    def monomial_words_forward(self,x):
        #The input x will be a 3-tensor of format num_samples x num_features_in x domain_dim
        #The idea is that x[i,:,:] is a MATRIX whose ROWS encode the various signals
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
        #################################################################################
        # IMPORTANT: WE USE THE FOLLOWING CONVENTION:
        # The input x will always be a 3-Tensor, whose components correspond to:
        # x[i,:,:] is the i-th data point matrix M
        # The rows M[a,:] of the matrix are the A features, each of which is a  
        # vector with num_vertices components
        # This convention is used so that one obtains matrix multiplication when thinking of block opeators
        # matching the usual linear algebra conventions.
        #################################################################################

        data_shape = x.shape 
        assert len(data_shape) == 3, "ERROR: Input must be a 3-tensor"
        assert self.monomial_word_support.operator_domain_dim == data_shape[2] and self.num_features_in == data_shape[1], "Evaluation point x[i,:,:] must be matrix of size num_features x domain dim"
        M = self.monomial_word_support
        answer_list = []
        for index in range(data_shape[0]):
            curr_data_point = x[index,:,:]
            evaluations_tensor = M.monomial_words_forward(curr_data_point)
            coefficients_tensor = self.coefficient_tensor
            #The following contraction defines the filter...
            res = torch.tensordot( coefficients_tensor, evaluations_tensor,dims = ([1,2],[0,2]))
            answer_list.append(res)
        return torch.stack(answer_list,0)


class NeuralReLuOperatorFilterLayer(OperatorFilterLayer):
    def __init__(self, num_features_in,num_features_out, monomial_word_support):        
        super().__init__(num_features_in,num_features_out, monomial_word_support)
    def forward(self,x):        
        z = super().forward(x) #See forward of the OperatorFilterLayer for details on conventions used
        relu = torch.nn.ReLU()
        return relu(z)



#TESTS:
global tol 
tol = 1e-5


def monomial_evaluation_test():
    t0 = torch.rand(2,2)
    t1 = torch.rand(2,2)
    operator_tuple = (t0,t1)
    M = MonomialWordSupport(num_variables=2, allowed_support = 4)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    X = M.variables
    new_monomial = X[0]*X[0]*X[1]*X[1]    
    mon_index = M.monomial_words.index(new_monomial)
    N2 = t0 @ t0 @ t1 @ t1 - M.operator_evaluated_monomial_words[mon_index]
    assert torch.norm(N2) < tol, "ERROR in evaluation"
    new_monomial = X[0]
    mon_index = M.monomial_words.index(new_monomial)
    N2 = t0 - M.operator_evaluated_monomial_words[mon_index]
    assert torch.norm(N2) < tol, "ERROR in evaluation"
    #TODO: Test the identity (the zeroth power)
    new_monomial = X[0]
    mon_index = M.monomial_words.index(new_monomial)
    N2 = t0 - M.operator_evaluated_monomial_words[mon_index]
    assert torch.norm(N2) < tol, "ERROR in evaluation"

def expression_evaluation_test():
    #This function tests the monomial evaluator
    t0 = torch.rand(2,2)
    t1 = torch.rand(2,2)
    t2 = torch.rand(2,2)
    operator_tuple = (t0,t1,t2)
    A = sp.symbols("x y z")
    expressions_list = [A[0], A[1]*A[2], A[0]**0, A[1]**2, A[0]+A[1]**2, (A[0]*A[1])*A[2]]
    ambient_variables = [A[j] for j in range(3)]
    M = MonomialWordSupport(num_variables=3, allowed_support = 2)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)#Evaluation of the monomial support at the given op tuple.
    #Tests:
    word_operator = M.evaluate_expression_in_operator_tuple(word_expression = expressions_list[0], ambient_variables=ambient_variables)
    N = word_operator-t0
    assert torch.norm(N) < tol, "Error in computing single variable operators"

    word_operator = M.evaluate_expression_in_operator_tuple(word_expression = expressions_list[1], ambient_variables=ambient_variables)
    N = word_operator- t1 @ t2
    assert torch.norm(N) < tol, "Error in computating a product"

    word_operator = M.evaluate_expression_in_operator_tuple(word_expression = expressions_list[2], ambient_variables=ambient_variables)
    N = word_operator-torch.eye(2)
    assert torch.norm(N) < tol, "Error in computing constant"

    word_operator = M.evaluate_expression_in_operator_tuple(word_expression = expressions_list[3], ambient_variables=ambient_variables)
    N = word_operator-t1 @ t1
    assert torch.norm(N) < tol, "Error in computing powers"

    word_operator = M.evaluate_expression_in_operator_tuple(word_expression = expressions_list[4], ambient_variables=ambient_variables)
    N = word_operator - (t0 + t1 @ t1)
    assert torch.norm(N) < tol, "Error in computing complex sums"

    word_operator = M.evaluate_expression_in_operator_tuple(word_expression = expressions_list[5], ambient_variables=ambient_variables)
    N = word_operator - t0@ t1 @ t2
    assert torch.norm(N) < tol, "Error in computing complex products"

    #Finally a bespoke expression evaluation test for idiosyncratic operator networks
    arr1 = [[0.5, 0], [0, 0.5]]
    t1 = torch.Tensor(arr1)
    operator_tuple = (t1,)
    x = sp.Symbol("x")
    M = MonomialWordSupport(num_variables=1, allowed_support =[x])#this is the monomial X[0]
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    expressions_list = [x**2,1+x]
    #
    word_operator = M.evaluate_expression_in_operator_tuple(word_expression = expressions_list[0], ambient_variables=ambient_variables)
    N = word_operator-t1 @ t1
    assert torch.norm(N) < tol, "Error in computing single variable operators"

    word_operator = M.evaluate_expression_in_operator_tuple(word_expression = expressions_list[1], ambient_variables=ambient_variables)
    N = word_operator-(torch.eye(2)+ t1)
    assert torch.norm(N) < tol, "Error in computing single variable operators"


def operator_matrix_from_coeffs_test():
    #set up the support, 2 variables degree at most three
    M = MonomialWordSupport(num_variables=2, allowed_support = 3)
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
    M = MonomialWordSupport(num_variables=2, allowed_support = 3)
    #We define an operator tuple to evaluate the monomial support as follows,
    domain_dim = 2    
    arr1 = [[0.5, 0], [0, 0.5]]
    arr2 = [[0, 0.5], [0.5, 0.0]]
    #arr2 = [[0, 0.5, 3],[0, 0.5, 3] ]
    t1 = torch.Tensor(arr1)
    t2 = torch.Tensor(arr2)
    operator_tuple = (t2,t1)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)#Evaluation of the monomial support at the given op tuple.       
    #With the evaluated support, we build a filter layer with feature sizes 2,3
    num_features_in = 2
    num_features_out = 2
    filter_layer = OperatorFilterLayer(num_features_in = num_features_in, 
                                       num_features_out = num_features_out, 
                                       monomial_word_support = M)
    
    #We inspect the tensor of coefficients, 
    #This 3-tensor has format: number_features_out x number_features_in x num_coeffs 
    coefficients_tensor = filter_layer.coefficient_tensor
    H10 = M.operator_matrix_from_coeffs(coefficients_vector = coefficients_tensor[1,0,:])
    H11 = M.operator_matrix_from_coeffs(coefficients_vector = coefficients_tensor[1,1,:])

    #We create a random tensor to simulate input signal
    #This is a 3-tensor with format: number_samples x num_features_in x domain_dim
    #Intuitively each input is a matrix whose ROWS are the various signals on the graph
    #*** See forward in the operator filter layer class for details***  
    num_samples = 8
    x_tensor = torch.Tensor(num_samples,num_features_in,domain_dim) 
    nn.init.uniform_(x_tensor) #the two 2-diml features are the rows of x_tensor

    #We compute the output according to the layer's forward implementation
    y_tensor = filter_layer.forward(x_tensor)
    assert y_tensor.shape[0] == x_tensor.shape[0]
    target_row = y_tensor[0,1,:]

    #and according to the block formula
    computed_row = torch.mv(H10, x_tensor[0,0,:]) + torch.mv(H11, x_tensor[0,1,:]) 

    #and compare them
    diff = target_row-computed_row
    assert torch.norm(diff) < tol, "ERROR, formula fails to match to desireable accuracy"



if __name__ == "__main__":
    expression_evaluation_test()
    monomial_evaluation_test() #Verifies that the evaluation behaves correctly. TODO: Should be made into a unit test
    operator_matrix_from_coeffs_test() #Verifies that the evaluation of operators behaves correctly. TODO: Should be made into a unit test
    forward_layer_test() #Verifies that the contraction in the forward layer agrees with the simple block-formula

    #EXAMPLES:
    #One defines an operator tuple as follows
    arr1 = [[0.5, 0], [0, 0.5]]
    arr2 = [[0, 0.5], [0.5, 0.0]]
    t1 = torch.Tensor(arr1)
    t2 = torch.Tensor(arr2)
    operator_tuple = (t2,t1)
    M = MonomialWordSupport(num_variables=2, allowed_support = 3)
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    #With the evaluated monomial support object we can build layers. 
    #See the tests above for details on how these objects are instantiated and how they behave

    #EXAMPLE 2:
    #It is also possible to define a collection of monomials as support
    arr1 = [[0.5, 0], [0, 0.5]]
    t1 = torch.Tensor(arr1)
    operator_tuple = (t1,)
    x = sp.Symbol("x")
    M = MonomialWordSupport(num_variables=1, allowed_support =[x])#this is the monomial X[0]
    M.evaluate_at_operator_tuple(operator_tuple=operator_tuple)
    #M.monomial_words_forward(x) #This can be used for evaluation of monomials at a given input

