# %%
# Import relevant libraries 
import numpy as np
from numpy.random import default_rng
from processData import processData
import datasetLoading as dataset
from sklearn.model_selection import KFold
import copy

# Set a random state for repeatable results 
rng = default_rng(101)


 # %%
#Sigmoid activation function
def activationSig( Z ):
    return 1 / ( 1 + np.exp( -Z ) )      

# Calculate the Loss
def loss (X, y, W):
    N = X.shape[1]
    y_predict = np.sum( X * W, axis = 1) 
    y_predict = activationSig(np.array(y_predict)).T
    epsilon = 1e-10
    return -1 / N * np.sum (y * np.log(y_predict + epsilon) + (1 - y) * np.log(1 - y_predict) ) 

def lNorm (W):
    # The last feature will be the bias term 
    return np.sum(np.abs(W[1:]))


# %% 
class analysis:
    def __init__( self, X, y, W, limit):

        self.X = X
        self.y = y
        self.W = W
        

        y_predict = np.sum( self.X * self.W, axis = 1)
        self.y_predict = activationSig(np.array(y_predict)).T

        self.limit = np.float32(limit)
    
    def metrics(self):

        y = np.reshape( self.y, ( len( self.y ), 1 ) )
        
        y_pred = self.y_predict
        y_pred = np.array( [ 0 if  i < self.limit else 1 for i in y_pred ] )
        y_pred = np.reshape( y_pred, ( len( y ), 1 ) )

        truePositive = np.count_nonzero( (y + y_pred) == 2)
        falseNegative = np.count_nonzero( (y - y_pred) == 1)
        falsePositive = np.count_nonzero( (y - y_pred) == -1)

        if truePositive > 0:
            precision = truePositive / ( truePositive + falsePositive )
            recall = truePositive / ( truePositive + falseNegative )
            
            f1 = 2 * ( precision * recall ) / ( precision + recall )
            print('Task', i, ', F1 Score: ', np.round(f1 * 100, 2))
        
        else:
            f1 = 0
        
        error = ( falseNegative + falsePositive ) / len( y )

        print('Task', i, ', Percentage Error: ', np.round(error *100, 2))
    
        return f1, error


# %%
# Define key parameters
m = dataset.m # number of features
t = dataset.t # number of tasks
rawData = dataset.rawData
taskList = dataset.taskList
testData = dataset.testData


test_size = 0.25 # Training and testing split for data
e = 0.1
c =  0.1 

#%%
processedData = {}
data = {}

for i in taskList:
    # Process the raw data for model (scale, split the data, etc)
    processedData[i] = processData( rawData[i], m, test_size )

    # Create data for k-folding ;) 
    y = np.append(processedData[i].y_train, processedData[i].y_test)
    X = np.append(processedData[i].X_train, processedData[i].X_test, axis = 0 )
    X_test = (processedData[i].scaleAndBias(testData[i][:, :-1]))
    y_test = testData[i][:, -1]
    data[i] = np.append(X, np.reshape(y, [len(y), 1]), axis = 1)
    testData[i] = np.append(X_test, np.reshape(y_test, [len(y_test), 1]), axis = 1)

# Swap data around
# data2 = {}
# data2[0] = data[1]
# data2[1] = data[0]
# data = data2
# planes = ['B1', 'B2', 'A1', 'A2']

#%%

# This K-fold is for splitting the data by features such that only one feature window is considered at a time
windows = KFold(n_splits = 2)
Results = []

for i in taskList: 
    Window = 1

    for k, window in windows.split(range(m)):
        print('Window: ', Window)

        # Insert a bias term into each fold and keep the target
        windowDataRange = np.insert( np.append(window + 1, -1), 0, 0 )
        windowData = [ data[i][ : , windowDataRange]   for i in taskList ] 
        windowData_test = [ testData[i][ : , windowDataRange]   for i in taskList ] 

        # Generate a new array for results for each of the folds
        X_train = np.array(windowData[i][ :, :-1])                    
        y_train = np.array(windowData[i][ :, -1])
        
        X_test = np.array(windowData_test[i][ :, :-1])  
        y_test = np.array(windowData_test[i][ :, -1])
    
        # Initialise Parameters
        # Weight matrix
        W = np.zeros([windowDataRange.shape[0] - 1 ])
        J0 = loss(X_train, y_train, W)

        # Number of features
        m_window = window.shape[0]

        # Determine the high impact weight 
        all_loss_pos = []
        all_loss_neg = []
        indexTracker = []

        for j in range(m_window):
            W_i = W.copy()
            W_i[j] = W_i[j] + e
            iteration_loss = loss(X_train, y_train, W_i)
            all_loss_pos.append(iteration_loss)

            W_i = W.copy()
            W_i[j] = W_i[j] - e
            iteration_loss = loss(X_train, y_train, W_i)
            all_loss_neg.append(iteration_loss)
        
        if np.min(all_loss_pos) < np.min(all_loss_neg):
            W[np.argmin(all_loss_pos)] = e
            indexTracker.append(np.argmin(all_loss_pos))
        else: 
            W[np.argmin(all_loss_neg)] = -e
            indexTracker.append(np.argmin(all_loss_neg))

        # Empirical Loss
        J = loss(X_train, y_train, W)

        # Regularisation loss
        lnorm = lNorm(W)
        
        # Regularsation Parameter, lambda
        reg_param = (J0 - J) / lnorm

        # Total loss
        T = reg_param * lnorm

        # Total loss (used in iteration 1)
        Y = J + T

        Z = 0

        while reg_param > 0: # Should be 0
            # Try to take backward step
            all_loss_pos = []
            all_loss_neg = []

            for j in indexTracker:
                W_x = copy.deepcopy(W)
                W_i = W_x.copy()
                W_i[j] = W_i[j] - np.sign( W_i[j] ) * e
                iteration_loss = loss(X_train, y_train, W_i) #+ reg_param * lNorm(W_i, t, m_window)
                all_loss_pos.append(iteration_loss)                            

            # Update the weights with the lowest overall loss from +e and -e lists
                W_x[indexTracker[np.argmin(all_loss_pos)]] = W_x[indexTracker[np.argmin(all_loss_pos)]] - np.sign( W_i[j] ) * e
                    
            # Regularisation loss
            T_x = reg_param * lNorm(W_x)

            # Empirical Loss
            J_x = loss(X_train, y_train, W_x)

            # Total loss
            Y_x = J_x + T_x

            # This is the criteria for taking a backwards step.
            if (Y_x - Y) < -c:
                # If it is met then update the parameters
                W = W_x
                Z += 1
                Y = Y_x
                print('BACKWARD')
            
            # Otherwise take a forward step
            else: 
                # Take a forward step
                all_loss_pos = []
                all_loss_neg = []
                W_x = copy.deepcopy(W)

                for j in range(m_window + 1):
                    # Determine the loss for each feature block with +e
                    W_i = W_x.copy()
                    W_i[j] = W_i[j] + e
                    iteration_loss = loss(X_train, y_train, W_i)
                    all_loss_pos.append(iteration_loss)

                    # Determine the loss for each feature block with -e
                    W_i = W_x.copy()
                    W_i[j] = W_i[j] - e
                    iteration_loss = loss(X_train, y_train, W_i)
                    all_loss_neg.append(iteration_loss)

                # Select the lowest loss from all of the +e and -e. 
                # The weights with the lowest loss will be updated
                if np.min(all_loss_pos) < np.min(all_loss_neg):
                    W_x[np.argmin(all_loss_pos)] = W_x[np.argmin(all_loss_pos)] + e
                    J_x = np.min(all_loss_pos)

                else: 
                    W_x[np.argmin(all_loss_neg)] = W_x[np.argmin(all_loss_neg)] -e
                    J_x = np.min(all_loss_neg)

                # Determine the lnorm for the new set of weights
                lnorm_x = lNorm(W_x)

                # Determine if the regularisation parameter should be updated
                reg_param = np.min( (reg_param, (J - J_x) / (np.sqrt(t) * e ) ) )

                # Redefine the weight matrix, losses and iteration number
                W = W_x
                J = J_x
                lnorm = copy.deepcopy(lnorm_x)
                Y = J + reg_param * lnorm
                Z += 1

                # Redefine the indexTracker for the backward steps
                indexTracker = []
                for l in np.argwhere(W != 0):
                    indexTracker.append(l)
            
            if Z%10 == 0:
                print(Z)
                print(reg_param)

        f1, error = analysis(X_test, y_test, W, 0.5).metrics()
        results = [Window, e, c, Z, f1, error, W, i]
        Results.append(results)


        np.save('STL_Boosted_Sensitivity_Results_Final_3.npy', Results)
            
        Window += 1   

# %%
