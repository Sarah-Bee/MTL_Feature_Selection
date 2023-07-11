# Import relevant libraries 
import copy
import numpy as np
from numpy.lib.function_base import kaiser
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.model_selection import train_test_split

# Set a random state for repeatable results 
rng = default_rng(101)

#%%
class processData:
    # Class to hold all of the variables generated from the raw data (including the initial weights)
    def __init__(self, rawData, m, test_size):

        X_train, X_test, y_train, y_test = train_test_split(rawData[:, :-1], rawData[:, -1], test_size = test_size, random_state = 20)
        y_train = np.reshape(y_train, (len(y_train), 1))
        self.X_testUnscaled = X_test

        self.mean = np.mean(X_train, axis = 0)
        self.variance = np.var(X_train, dtype=np.float64)

        X_train = self.scaleAndBias(X_train)
        X_test = self.scaleAndBias(X_test)
       
        # Generate a set of weights
        W = rng.random((1, m + 1)) * 0.01

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.W = W  


    def scaleAndBias (self, data):
       
        scaledData = (data - self.mean) / self.variance
        length = len(scaledData)
        return np.append(np.reshape(np.ones(length), (length, 1)), scaledData, axis = 1)

# %%