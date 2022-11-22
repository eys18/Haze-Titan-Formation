#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:27:36 2020

@author: ewansaw
"""
import numpy as np
import matplotlib.pyplot as plt

#%%NODES vs error
num_nodes = np.array([20, 50, 100, 200, 1000])
nodes3_mape = np.array([95.2, 80.1, 85, 62.9, 65.3])
nodes3_mape_err = np.array([27.9, 13.4, 15.4, 7.1, 8.8])

nodes3_mse = np.array([1.70, 1.58, 1.52, 1.54, np.float("nan")])
nodes3_mse_err = np.array([0.11, 0.13, 0.07, 0.07, np.float("nan")]) 

nodes5_mape = np.array([67.225, 68.9, 65.43, 64.82, 71.48])
nodes5_mape_err = np.array([7.074, 7.80, 8.15, 15.31, 15.81])

nodes5_mse = np.array([1.40, 1.27, 1.25, 1.20, np.float("nan")])
nodes5_mse_err = np.array([0.09, 0.16, 0.08, 0.04, np.float("nan")])


fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('Error against number of nodes ')
ax1.errorbar(num_nodes, nodes3_mape, yerr = nodes3_mape_err, label = "3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)
ax1.errorbar(num_nodes, nodes5_mape, yerr = nodes5_mape_err, label = "5 Inputs", 
             fmt = 'o', capsize = 3)
ax2.errorbar(num_nodes, nodes3_mse, yerr = nodes3_mse_err, label = "3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)
ax2.errorbar(num_nodes, nodes5_mse, yerr = nodes5_mse_err, label = "5 Inputs", 
             fmt = 'o', capsize = 3)
ax1.legend()
ax2.legend()
fig.text(0.5, 0.04, "Number of Nodes", ha='center', va='center')
fig.text(0.06, 0.7, "MAP error", ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.3, "RMS error", ha='center', va='center', rotation='vertical')

#%% LAYERS VS ERROR
num_layers = np.array([1, 2, 3, 4, 5])
layers3_mape = np.array([133.2, 92.5, 89.6, 80.1, 80.0])
layers3_mape_err = np.array([33.8, 30.1, 38.8, 13.4, 12.6])

layers3_mse = np.array([2.06, 1.66, 1.80, 1.49, 1.46])
layers3_mse_err = np.array([0.45, 0.19, 0.33, 0.04, 0.08])

layers5_mape = np.array([82.52, 68.27, 65.46, 65.96, 64.97])
layers5_mape_err =  np.array([24.27, 8.91, 10.75, 2.16, 9.68])

layers5_mse = np.array([1.58, 1.31, 1.31, 1.29, 1.22])
layers5_mse_err = np.array([0.13, 0.09, 0.09, 0.05, 0.05])


fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('Error against number of layers ')
ax1.errorbar(num_layers, layers3_mape, yerr = layers3_mape_err, label = "3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)
ax1.errorbar(num_layers, layers5_mape, yerr = layers5_mape_err, label = "5 Inputs", 
             fmt = 'o', capsize = 3)
ax2.errorbar(num_layers, layers3_mse, yerr = layers3_mse_err, label = "3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)
ax2.errorbar(num_layers, layers5_mse, yerr = layers5_mse_err, label = "5 Inputs", 
             fmt = 'o', capsize = 3)
ax1.legend()
ax2.legend()
fig.text(0.5, 0.04, "Number of Layers", ha='center', va='center')
fig.text(0.06, 0.7, "MAP error", ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.3, "RMS error", ha='center', va='center', rotation='vertical')



#%%#LOSS FUNC VS ERROR
loss_func = ["Mae", "Mse", "MSle"]
loss3_mape = [89.1, 74.3, 75.0]
loss3_mape_err = [13.4, 11.2, 17.2]

loss3_mse = [1.49, 1.57, 2.38]
loss3_mse_err = [0.04, 0.05, 0.09]

loss5_mape = [68.90, 81.56, 70.6]
loss5_mape_err = [7.80, 13.00, 15.3]

loss5_mse = [1.27, 1.39, 1.91]
loss5_mse_err = [0.16, 0.09, 1.17]


fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('Error against loss function ')
ax1.errorbar(loss_func, loss3_mape, yerr = loss3_mape_err, label = "3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)
ax1.errorbar(loss_func, loss5_mape, yerr = loss5_mape_err, label = "5 Inputs", 
             fmt = 'o', capsize = 3)
ax2.errorbar(loss_func, loss3_mse, yerr = loss3_mse_err, label = "3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)
ax2.errorbar(loss_func, loss5_mse, yerr = loss5_mse_err, label = "5 Inputs", 
             fmt = 'o', capsize = 3)
ax1.legend()
ax2.legend()
fig.text(0.5, 0.04, "Loss Function", ha='center', va='center')
fig.text(0.06, 0.7, "MAP error", ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.3, "RMS error", ha='center', va='center', rotation='vertical')

#%% #OPTIMISER VS ERROR
optimisers = ['Adam', 'Adagrad', 'RMSprop', 'Adadelta']
optimiser3_mape = [74.1, 102.4, 74.3, 137.2]
optimiser3_mape_err = [14.9, 27.7, 17.3, 15.3]
 
optimiser3_mse =[1.51, 2.08, 1.40, 2.49]
optimiser3_mse_err = [0.1, 0.43, 0.07, 0.15]
 
optimiser5_mape = [68.9, 72.35, 82.89, 67.66]
optimiser5_mape_err = [7.8, 12.12, 11.81, 15.26]

optimiser5_mse = [1.27,1.38,1.21,1.26]
optimiser5_mse_err = [0.16,0.08,0.07,0.04]


fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('Error against optimiser ')
ax1.errorbar(optimisers, optimiser3_mape, yerr = optimiser3_mape_err, label = "3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)
ax1.errorbar(optimisers, optimiser5_mape, yerr = optimiser5_mape_err, label = "5 Inputs", 
             fmt = 'o', capsize = 3)
ax2.errorbar(optimisers, optimiser3_mse, yerr = optimiser3_mse_err, label ="3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)
ax2.errorbar(optimisers, optimiser5_mse, yerr = optimiser5_mse_err, label = "5 Inputs", 
             fmt = 'o', capsize = 3)
ax1.legend()
ax2.legend()
fig.text(0.5, 0.04, "Optimiser", ha='center', va='center')
fig.text(0.06, 0.7, "MAP error", ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.3, "RMS error", ha='center', va='center', rotation='vertical')


#%%#ACTIVATION FUNC VS ERROR
activation = ['Relu', 'Selu', 'Softmax', 'Sigmoid']
activation3_mape = [74.1, 69.7, 117.1, 67.9]
activation3_mape_err =[14.9, 8.6, 9.8, 13.3]

activation3_mse = [1.51,1.41,2.15,1.43]
activation3_mse_err = [0.1,0.04,0.19,0.08]

activation5_mape = [68.9, 59.33, 64.24, 66.76]
activation5_mape_err = [7.8, 9.72, 5.81, 8.42] 

activation5_mse = [1.27, 1.23, 1.3, 1.25]
activation5_mse_err = [0.16, 0.23, 0.05, 0.11]

fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('Error against activation function ')
ax1.errorbar(activation, activation3_mape, yerr = activation3_mape_err, label = "3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)
ax1.errorbar(activation, activation5_mape, yerr = activation5_mape_err, label = "5 Inputs", 
             fmt = 'o', capsize = 3)
ax2.errorbar(activation, activation3_mse, yerr = activation3_mse_err, label = "3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)
ax2.errorbar(activation, activation5_mse, yerr = activation5_mse_err, label = "5 Inputs", 
             fmt = 'o', capsize = 3)
ax1.legend()
ax2.legend()
fig.text(0.5, 0.04, "Activation Function", ha='center', va='center')
fig.text(0.06, 0.7, "MAP error", ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.3, "RMS error", ha='center', va='center', rotation='vertical')

#%% #Type of layer vs error 

layer = ['Conv1D + LSTM layers', 'Only LSTM layer', 'Only Conv1D']


layer3_mape = [54.85, 51.24, 128.29]
layer3_mape_err = [9.87, 6.84, 60.05]

layer3_mse = [1.78, 1.59, 1.47]
layer3_mse_err = [0.25, 0.17, 0.19]

fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('Error against type of layer for 3 inputs')
ax1.errorbar(layer, layer3_mape, yerr = layer3_mape_err, label = "3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)

ax2.errorbar(layer, layer3_mse, yerr = layer3_mse_err, label ="3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)

fig.text(0.5, 0.04, "Layers", ha='center', va='center')
fig.text(0.06, 0.7, "MAP error", ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.3, "RMS error", ha='center', va='center', rotation='vertical')






#%%

epochs = [20, 50, 100, 200, 500, 1000, 1500, 2000]


epoch3_mape = [136.30970614047664, 97.76737804763846, 69.48866559956917, 53.75269681578103, 
               33.3000649524109, 23.58961144756779, 20.795883570259928, 22.940165377813123]
epoch3_mape_err = [30.24595223725732, 29.240190651517814, 10.51364339534313, 9.374146314276855,
                   8.621434895898572, 3.620981665801645, 7.558342675449468, 3.9922492138504877]

epoch3_mse = [1.7545949954466522, 1.6815009921250723, 1.434605267806396, 1.1828250360864825,
              0.9443026400308557, 0.6808053459985042, 0.6215173279354598, 0.6906459430706844]
epoch3_mse_err = [0.18662600917524433, 0.2760644028702776, 0.21151202002221808, 0.047150328738861574,
                  0.11870710843140123, 0.09622062920175649, 0.12614136393140699, 0.13010865990203024]

fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('Error against number of epochs for 3 inputs')
ax1.errorbar(epochs, epoch3_mape, yerr = epoch3_mape_err, label = "3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)


ax2.errorbar(epochs, epoch3_mse, yerr = epoch3_mse_err, label ="3 Inputs (spatial)", 
             fmt = 'o', capsize = 3)

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')

fig.text(0.5, 0.04, "Number of Epochs", ha='center', va='center')
fig.text(0.06, 0.7, "MAP error", ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.3, "RMS error", ha='center', va='center', rotation='vertical')

#plt.xscale("log")


'''
20 epochs
mape = [136.30970614047664, 30.24595223725732]
mse = [1.7545949954466522, 0.18662600917524433]

100 epochs
mape = [69.48866559956917, 10.51364339534313]
mse = [1.434605267806396, 0.21151202002221808]

200 epochs
mape = [53.75269681578103, 9.374146314276855]
mse = [1.1828250360864825, 0.047150328738861574]

500 epochs 
mape =  [33.3000649524109, 8.621434895898572]
mse = [0.9443026400308557, 0.11870710843140123]

1000 epochs
mape = [23.58961144756779, 3.620981665801645]
mse = [0.6808053459985042, 0.09622062920175649]

2000 epochs
mape = [22.940165377813123, 3.9922492138504877]
mse = [0.6906459430706844, 0.13010865990203024]
'''

#%%

inputs = [3, 5]


mape = [21.383423261535405, 19.20941132021589]
        
mape_err = [2.739440038079057, 6.3944828325357745]

mse = [0.7463070368205182, 0.5404141836541076]
       
mse_err = [0.10951614229770174, 0.22474344463718296]

    
fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('Error against number of Inputs')
ax1.errorbar(inputs, mape, yerr = mape_err, fmt = 'o', capsize = 3)

ax2.errorbar(inputs, mse, yerr = mse_err, fmt = 'o', capsize = 3)

ax1.set_xlim([0, 8])
ax2.set_xlim([0, 8])
fig.text(0.5, 0.04, "Number of Inputs", ha='center', va='center')
fig.text(0.06, 0.7, "MAP error", ha='center', va='center', rotation='vertical')
fig.text(0.06, 0.3, "RMS error", ha='center', va='center', rotation='vertical')
