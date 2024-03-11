#This script loads the movieLens databases and organizes the data in batches and tensors
#appropriately. It also builds the similarity graphs and their shift operators

from movie_data_process import movie_lens, movie_dataset

# TODO BERNIE: Check this with Mauri.
def build_operator_tuple(num_vertices):
    # What should I do with num_vertices? Perhaps we should select this number of movies from orig dataset?
    return movie_dataset.process_movie_dataset()
    # returns a PAIR of operators (num_vertices x num_vertices matrices)
    # see data.build_operator_tuple for details on expected output
    #return t0, t1


def get_batch(num_vertices,n_samples):
    """ The structure of the returned training samples xtrain, ytrain is the following:
        (1) x[i,j,k] is a 3-tensor such that x[i,:,:] is a matrix (actually a row_vector) 
        of format  1 x num_vertices containing the score of all num_vertices movies for 
        person i (more precisely x[i,0,j] = rating of movie j for person i)
        (2) y[i,j] is a matrix (more precisely a column vector) where y[i,:]=y[i,0] is a 
        scalar, the rating given for the STARWARS movie by person i.
        """
    pass
    #return xtrain, ytrain