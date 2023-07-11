#%%
import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt, cm

rng = default_rng(10)

# Choose which dataset to load / generate
dataset = 'Engineering' # Select from ['Gaussian', 'Engineering']
   
# %%
if dataset == 'Engineering':
    import scipy.stats as stats

    n = 750 # number of sampled data points per class [50, 100]
    n_test = int( n / 3 ) # 25% of the total data
    t = 3 # Need to change this depending on the task being completed
    taskList = np.arange(t)
    planes = ['A1', 'B1', 'A2', 'B2', 'C1', 'C2']

    data_test = {}
    data = {}
    var = {}


    for plane in planes:
        data_pre_path = './Tailplane_Data/' # file path
        path = data_pre_path + plane + '_f.npy'
        data[ plane ] = np.load(path)

        mean = data[ plane ][:, 2:].mean( axis = 0 )
        sigma = data[ plane ][:, 2:].std( axis = 0 )
        var[plane] = sigma

        data[plane] = stats.multivariate_normal.rvs(mean, np.diag(sigma), n, random_state = 101) #101
        data_test[plane] = stats.multivariate_normal.rvs(mean, np.diag(sigma), n_test, random_state = 101) #101

        n, m = data[plane].shape

        if planes.index(plane)%2 == 1:
            classID = np.ones([n, 1])
            classID_test = np.ones([n_test, 1])

        elif planes.index(plane)%2 == 0:
            classID = np.zeros([n, 1])
            classID_test = np.zeros([n_test, 1])

        data[plane] = np.concatenate( ( data[plane], classID ), axis = 1 )
        data_test[plane] = np.concatenate( ( data_test[plane], classID_test ), axis = 1 )


    rawData = {}
    testData = {}
    
    for task in taskList: 
        class1 = data [ list(data.keys())[task * 2] ]
        class2 = data [ list(data.keys())[task * 2 + 1] ] 

        class1_test = data_test [ list(data_test.keys())[task * 2] ]
        class2_test = data_test [ list(data_test.keys())[task * 2 + 1] ] 

        rawData[task] = np.concatenate( ( class1, class2 ) )
        testData[task] = np.concatenate( ( class1_test, class2_test ) )

    n = n * 2



# %%
if dataset == 'Gaussian':

    # Define key parameters
    n = 200 # number of samples per class (2 classes per task)
    m = 10 # number of features
    t = 2 # number of tasks
    taskList = np.arange(t)

    # Gaussian Data
    def multiVariateGaussian(m, n, target):
        # Define the mean and variance for each feature
        mean = rng.normal(rng.random(), rng.random(), m)
        variance = rng.normal(rng.random(), rng.random(), m) * np.eye(m)
        # Define a vector for the target variable for binary classification [0,1]
        if target == 0:
            target = np.reshape(np.zeros(n), (n, 1))
        elif target == 1:
            target = np.reshape(np.ones(n), (n, 1))
        # Generate samples from the mean and variance and append the target
        return np.append(rng.multivariate_normal(mean, variance, n), target, axis = 1)

    # Define a new data set for each of the tasks
    rawData = {}
    for i in taskList:
        class1 = multiVariateGaussian(m, n, 0)
        class2 = multiVariateGaussian(m, n, 1)
        rawData[i] = np.concatenate((class1, class2))

