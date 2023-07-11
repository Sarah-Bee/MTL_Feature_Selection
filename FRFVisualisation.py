#%%
# Import relevant libraries 
import numpy as np
from matplotlib import pyplot as plt, cm
from numpy.random import default_rng
import datasetLoading as dataset

# Set a random state for repeatable results 
rng = default_rng(101)

#%%
# Data Loading
# Planes for analysis
#planes = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'] 
planes = dataset.planes

# Frequencies for x-axis
freq =   np.load('./Tailplane_Data/freq.npy', allow_pickle = True)[()]

# Overall useful range of the FRF (frequencies analysed in algorithm)
clip = np.arange(107, 695)  

# Intial FRF data
FRFData =   np.load('./Tailplane_Data/FRF.npy', allow_pickle = True)[()]

# Truncate FRF data to match analysis
for plane in planes:
    FRFData[plane] = FRFData[plane][7:-5]

# Determine the number of tasks
tasks = dataset.t # number of tasks

# Define the "raw data" as used in the algorithm 
rawData = dataset.rawData

# Define variance data
varData = dataset.var

# Change the [0, 1] classification to [2, 3] to differentiate between task 1 and task 2
for i in range(len(rawData[1][:, -1])):
    if rawData[1][i, -1] == 0:
        rawData[1][i, -1] = 2
    else:
        rawData[1][i, -1] = 3

for i in range(len(rawData[2][:, -1])):
    if rawData[2][i, -1] == 0:
        rawData[2][i, -1] = 4
    else:
        rawData[2][i, -1] = 5



# Define the sampled data
dataI  = np.append(rawData[0], rawData[1], axis = 0)
data = np.append(dataI, rawData[2], axis = 0)

#%%
# FRF with 1SD and all points for A and B
# Colour palettes for graph:
cmap = ['winter', 'cool', 'spring']
cmapW = cm.get_cmap('winter')
cmapA = cm.get_cmap('cool')
bmap1 = [cmapW(0.0), 'g', 'deepskyblue', 'purple', 'dodgerblue', 'fuchsia']
bmap = [cmapW(0.0), 'seagreen', cmapA(0.2), cmapA(0.7), 'dodgerblue', 'violet']

# # Font size
# plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'font.size': 100})


# # Linestyles for graph:
linestyle = ['-', '-']
marker = ['*', 'H']
linewidth = 20


# %%
# Font size
plt.rcParams.update({'font.size': 100})

linewidth = 20
fig, axs = plt.subplots(3, 1, figsize=[120, 50], sharex = 'row', sharey = 'row')

planes = ['A1', 'B1', 'A2', 'B2', 'C1', 'B1']
i = 0

for plane in planes:
        index = planes.index(plane)
        xx = FRFData[plane]

        yy_var = np.std( data[data[:, -1] == index] , axis = 0)[:-1]
        yy_var = np.log(yy_var) / xx.std()

        yy = np.log(np.absolute(np.mean( data[data[:, -1] == index] , axis = 0)[:-1]))

        yy_up = (yy - xx.mean())/xx.std() + yy_var
        yy_down = (yy - xx.mean())/xx.std() - yy_var

        if i == 5:
            axs[-1].fill_between(freq[1:-1], yy_up, yy_down, 
            alpha = 0.4, color = bmap[index])

            axs[-1].plot(freq[1:-1], (xx - xx.mean())/xx.std(), 
            c=bmap1[index], linestyle = linestyle[index % 2], linewidth=linewidth, label=planes[index])

        else:
            axs[int(np.floor(index/2))].fill_between(freq[1:-1], yy_up, yy_down, 
            alpha = 0.4, color = bmap[index])

            axs[int(np.floor(index/2))].plot(freq[1:-1], (xx - xx.mean())/xx.std(), 
            c=bmap1[index], linestyle = linestyle[index % 2], linewidth=linewidth, label=planes[index])
        
        i += 1


# Set axes limits (will be the same for all subplots)
[ax.set_xlim(freq[1:-1].min(), freq[1:-1].max()) for ax in axs]
[ax.set_ylim((((xx - xx.mean())/xx.std()).min() - 1.5, ((xx - xx.mean())/xx.std()).max() + 1.5)) for ax in axs]

# Set y-axis label
axs[-2].set_ylabel('Normalised Response')
[ax.legend(loc='lower right', frameon=True) for ax in axs]
axs[-1].set_xlabel('Frequency (Hz)') 

fig.suptitle('FRF for the Three Tasks')


fig.savefig('demo3.png', transparent=True)
# %%
