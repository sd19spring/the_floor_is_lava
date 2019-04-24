import numpy as np
from matplotlib import pyplot as plt
"""

Demo code to create a heatmap plot from a matrix.
Requires: matplotlib and numpy

Takes a 5x5 matrix of dummy data and creates a heatmap using matplotlib

@author: Gabriella Bourdon, Michael Remley

"""

#initalize matrix 5 rows by 5 columns
arry = np.zeros((5,5))

def get_person_data():
    #get data from person tracking
    pass

def to_data_matrix(sample_list):
    """
    Wrapper for numpy stack method
    Takes a matrix as a list of lists and converts into numpy matrix

    Inputs: List of n elements with n elements each
    Outputs: Numpy matrix
    Returns: Numpy matrix
    """
    matrix = np.stack(sample_list)
    print (matrix)
    return (matrix)

def draw_heatmap(data_matrix):
    """
    Plots heatmap of a data matrix

    Inputs: data_matrix
    Outputs: Matplotlib plot
    Returns: none
    """
    plt.imshow(data_matrix, cmap='hot', interpolation='nearest')
    plt.show()
    # Optional, save figure to file
    #plt.savefig('matrix.jpg')


if __name__ == '__main__':
    # Make dummy data for the test
    dummy_list = [[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5]]
    # Convert to numpy matrix
    dummy_matrix  = to_data_matrix(dummy_list)
    # Make a heatmap of dummy data
    draw_heatmap(dummy_matrix)
