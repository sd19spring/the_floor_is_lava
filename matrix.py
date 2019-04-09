import numpy as np
import matplotlib.pyplot as plt

#initalize matrix 5 rows by 5 columns
arry = np.zeros((5,5))

def get_person_data():
    #get data from person tracking
    pass

def data_matrix(sample_list):
    #i is rows, j is columns. transform imported data into a matrix
    ix = 0
    for i in range(5):
        for j in range(5):
            arry[i] = ix
            ix+=1
    a = np.stack(dummy_list)
    print (a)
    return (a)

def test_heatmap(a):
    #create a heatmap from 2D matrix
    a = np.stack(dummy_list)
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()


if __name__ == '__main__':

    dummy_list = [[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5]]
    data_matrix(dummy_list)
    test_heatmap(dummy_list)

# #option 2
# def data_matrix():
#     np.zeros((2,2))
#     dummy_list = [[1,2,3,4,5],[5,4,3,2,1]]
#     a = np.array(dummy_list)
#     print (a);
# data_matrix()
