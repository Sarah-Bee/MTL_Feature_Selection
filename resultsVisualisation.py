#%%
# Import relevant libraries 
import numpy as np
from matplotlib import pyplot as plt, cm
import matplotlib.gridspec as gridspec
from numpy.random import default_rng
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import datasetLoading as dataset
from LASSO_Classes import processData
from sklearn.model_selection import KFold
import copy

# Set a random state for repeatable results 
rng = default_rng(101)

#%%
# FRF DATA
FRFData =   np.load('FRF.npy', allow_pickle = True)[()]
freq =   np.load('freq.npy', allow_pickle = True)[()]
planes = ['A1', 'B1', 'A2', 'B2'] 

for plane in planes:
    FRFData[plane] = FRFData[plane][7:-5]

# PCA DATA
tasks = dataset.t # number of tasks
features = dataset.m # number of features
rawData = dataset.rawData

pca = PCA()
processedData = {}

for i in range(tasks):
    # Process the raw data for model (scale, split the data, etc)
    processedData[i] = processData( rawData[i], features, 0.0001 )

    # Add the test and train togehter (project all of the data)
    target = np.append(processedData[i].y_train, processedData[i].y_test)
    dataAll = np.append(processedData[i].X_train, processedData[i].X_test, axis = 0) 

# Change the [0, 1] classification to [2, 3] to differentiate between task 1 and task 2
for i in range(len(rawData[1][:, -1])):
    if rawData[1][i, -1] == 0:
        rawData[1][i, -1] = '2'
    else:
        rawData[1][i, -1] = '3'

for i in range(len(rawData[2][:, -1])):
    if rawData[2][i, -1] == 0:
        rawData[2][i, -1] = '4'
    else:
        rawData[2][i, -1] = '5'

# Define the sampled data
dataV  = np.append(rawData[0], rawData[1], axis = 0)
#clip = np.arange(107, 695)  # Overall useful range of the FRF


# %%
fileName = 'STL_Boosted_Sensitivity_Results_Final_3.npy'
# Load file 
result =  np.load(fileName, allow_pickle = True)
plt.rcParams.update({'font.size': 100})
df = pd.DataFrame(result)
df.columns = ['Window', 'Epsilon', 'Chi', 'Iterations', 'F1_Score', 'Error', 'Weights', 'Task']

weightMatrix = [np.array(x) for x in df['Weights'] ]
weightMatrix = np.array( weightMatrix )


# Gini Index!
weightsGI = []

for weight in df['Weights']:
    value = np.sort( np.abs ( weight ) ) 
    weightsGI.append( value )

df['Weights_GI'] = weightsGI
GIList = []

for weight in df['Weights_GI']:
    M = len(weight)
    ell1 = np.sum(weight)
    if ell1 == 0:
        GI = 1
    
    else: 
        k = np.arange(len(weight))
        sumk = (M - k + 0.5) / M
        sumTotal = np.sum( weight * sumk )
        GI = 1 - (2 / ell1 * sumTotal)

    GIList.append(GI)

df['GI'] = GIList

for i in range ( 2 * 2):
    for x in np.arange(weightMatrix[i].shape[0]):
        if weightMatrix[i][x] == 0:
            weightMatrix[i][x] = np.nan

vmin = np.nanmin(weightMatrix)
vmax = np.nanmax(weightMatrix)

cmap = [copy.copy(cm.get_cmap("winter")), copy.copy(cm.get_cmap("cool"))]
[x.set_bad('white') for x in cmap ]

a = [0 , 2]
b = [2,  None]
axs = [0] * 2
cmapW = cm.get_cmap('winter')
#bmap = [cmapW(0.0), 'seagreen', 'purple', 'deepskyblue'] # peru
#bmap = [cmapW(0.0), 'seagreen', cmapA(0.0), 'peru']
bmap = [cmapW(0.0), 'g', 'deepskyblue', 'purple']
linestyle = ['-', '-']

bigFig, axs = plt.subplots(2, 2, figsize=[120, 60], sharex = 'row', sharey = 'row')

bigFig.subplots_adjust(wspace=0, top = 0.9)

gs1 = axs[0, 0].get_gridspec()
gs2 = axs[1, 0].get_gridspec()
gs1.update(wspace = 0)
# remove the underlying axes

axFRF1 = bigFig.add_subplot(gs1[0, :])
axFRF2 = bigFig.add_subplot(gs1[1, :])

bigFig.tight_layout()

FRF = [axFRF1, axFRF2]
bar = [0.9, 0.4]

for j in [0,1]:
    for i in range(2):
        weightsMatrixTask = weightMatrix[j * 2 + i]
        weightsMatrixTask = np.reshape(weightsMatrixTask, (1, len(weightsMatrixTask)) )


        im = axs[j][i].imshow(weightsMatrixTask,  aspect='auto', vmax = vmax, vmin = vmin, 
            cmap = cmap[j], interpolation = 'none', alpha = 0.5)
        
        labelF1 = np.round(100 * np.array(df['F1_Score'])[2*j + i], 1)
        labelGI = np.round(np.array(df['GI'])[2*j + i], 2)

        axs[j][i].text(100, 0.41, 'F1 Score: ' + str(labelF1) + '\nGini Index: ' + str(labelGI), 
            ha = 'center', size=120, bbox=dict(boxstyle="round,pad=0.3", fc = (1, 1, 1, 0.9), ec = 'black', lw = 2))
        
        axs[j][i].get_xaxis().set_visible(False)
        axs[j][i].get_yaxis().set_visible(False)
    
    cax = bigFig.add_axes([0.18, bar[j], 0.2, 0.02])
    plt.colorbar(im, cax = cax, shrink = 0.8, orientation='horizontal')
    axs[j][0].spines['right'].set_linewidth(50)
    axs[j][0].spines['right'].set_linestyle('--')

    

    FRF[j].patch.set_alpha(0)

    for plane in planes[a[j]:b[j]]:
        index = planes.index(plane)
        xx = FRFData[plane]
        FRF[j].plot(freq[1:-1], (xx - xx.mean())/xx.std(), 
                c=bmap[index], linestyle = linestyle[index % 2], linewidth=20, label=planes[index])

    FRF[j].set_xlim((freq[1], freq[-1]))
    FRF[j].set_ylim((((xx - xx.mean())/xx.std()).min() - 1.5, ((xx - xx.mean())/xx.std()).max() + 1.5))
    FRF[j].xaxis.set_ticks_position('top')
    FRF[j].set_xlabel('Frequency (Hz)', fontsize=120) 
    FRF[j].xaxis.set_label_position('top') 
    FRF[j].set_ylabel('Normalised Response', fontsize=150)
    FRF[j].legend(loc='upper left', fontsize=150)
    FRF[j].title.set_text('Task'+ str(j+1))

plt.show()
bigFig.savefig('STLFinalResults_2Windows.png', bbox_inches='tight')


# %%
fileName = 'MTL_Boosted_Sensitivity_Results_Final_5.npy'
# Load file 
result =  np.load(fileName, allow_pickle = True)

df = pd.DataFrame(result)
df.columns = ['Window', 'Epsilon', 'Chi', 'Iterations', 'F1_Score', 'Error', 'Weights']

weights = np.array(df['Weights'])
weightsArray = [np.array(weights[i]) for i in range( len(weights) ) ]
weightsArray = np.array(weightsArray)

df0 = df.copy()
df0['F1_Score'] = df['F1_Score'].apply(pd.Series)[0]
df0['Error'] = df['Error'].apply(pd.Series)[0]
df0['Weights']= [list(weightsArray[i, 0, :]) for i in range(len(weightsArray))]
df0['Task'] = 1 * np.ones(len (df0))

df1 = df.copy()
df1['F1_Score'] = df['F1_Score'].apply(pd.Series)[1]
df1['Error'] = df['Error'].apply(pd.Series)[1]
df1['Weights'] = [list(weightsArray[i, 1, :]) for i in range(len(weightsArray))]
df1['Task'] = 2 * np.ones(len (df1))

df = pd.concat([df0, df1])

# Gini Index!
weightsGI = []

for weight in df['Weights']:
    value = np.sort( np.abs ( weight ) ) 
    weightsGI.append( value )

df['Weights_GI'] = weightsGI
GIList = []

for weight in df['Weights_GI']:
    M = len(weight)
    ell1 = np.sum(weight)
    if ell1 == 0:
        GI = 1
    
    else: 
        k = np.arange(len(weight))
        sumk = (M - k + 0.5) / M
        sumTotal = np.sum( weight * sumk )
        GI = 1 - (2 / ell1 * sumTotal)

    GIList.append(GI)

df['GI'] = GIList

weightMatrix = np.array( df['Weights'] )
weightMatrix = [ np.array(x) for x in weightMatrix]

for i in range(2):
    for x in np.arange(weightMatrix[i].shape[0]):

        if weightMatrix[i][x] == 0:
            weightMatrix[i][x] = np.nan


weightMatrix = np.array(weightMatrix)

bigFig, axs = plt.subplots(2, 2, figsize=[120, 60], sharex = 'row', sharey = 'row')

bigFig.subplots_adjust(wspace=0)

gs1 = axs[0, 0].get_gridspec()
gs2 = axs[1, 0].get_gridspec()
gs1.update(wspace = 0)
# remove the underlying axes

axFRF1 = bigFig.add_subplot(gs1[0, :])
axFRF2 = bigFig.add_subplot(gs1[1, :])

bigFig.tight_layout()

FRF = [axFRF1, axFRF2]
bar = [0.9, 0.4]

for j in [0,1]:
    for i in range(2):
        weightsMatrixTask = np.reshape(weightMatrix[i], [1, len(weightMatrix[i])] )
        

        im = axs[j][i].imshow(weightsMatrixTask,  aspect='auto', vmax = vmax, vmin = vmin, 
            cmap = cmap[j], interpolation = 'none', alpha = 0.5)
        
        labelF1 = np.round(100 * np.array(df['F1_Score'])[2*j + i], 1)
        labelGI = np.round(np.array(df['GI'])[2*j + i], 2)

        axs[j][i].text(100, 0.41, 'F1 Score: ' + str(labelF1) + '\nGini Index: ' + str(labelGI), 
            ha = 'center', size=120, bbox=dict(boxstyle="round,pad=0.2", fc = 'white', ec = 'black', lw = 2))
        
        axs[j][i].get_xaxis().set_visible(False)
        axs[j][i].get_yaxis().set_visible(False)
    
    cax = bigFig.add_axes([0.18, bar[j], 0.2, 0.02])
    plt.colorbar(im, cax = cax, shrink = 0.8, orientation='horizontal')
    axs[j][0].spines['right'].set_linewidth(50)
    axs[j][0].spines['right'].set_linestyle('--')

    

    FRF[j].patch.set_alpha(0)

    for plane in planes[a[j]:b[j]]:
        index = planes.index(plane)
        xx = FRFData[plane]
        FRF[j].plot(freq[1:-1], (xx - xx.mean())/xx.std(), 
                c=bmap[index], linestyle = linestyle[index % 2], linewidth=20, label=planes[index])

    FRF[j].set_xlim((freq[1], freq[-1]))
    FRF[j].set_ylim((((xx - xx.mean())/xx.std()).min() - 1.5, ((xx - xx.mean())/xx.std()).max() + 1.5))
    FRF[j].xaxis.set_ticks_position('top')
    FRF[j].set_xlabel('Frequency (Hz)', fontsize=120) 
    FRF[j].xaxis.set_label_position('top') 
    FRF[j].set_ylabel('Normalised Response', fontsize=150)
    FRF[j].legend(loc='upper left', fontsize=150)
    FRF[j].title.set_text('Task'+ str(j+1))

plt.show()
bigFig.savefig('MTLFinalResults_2Windows.png', bbox_inches='tight')
# %%
