# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:18:48 2020
 
@author: Rayan and Ewan
"""
from sklearn.metrics import mean_squared_error
from math import sqrt
#import imblearn
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
import pandas as pd
import numpy as np
import sklearn as sk
from numpy import array
import matplotlib.pyplot as plt
from numpy import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
#from imblearn.pipeline import Pipeline
#from tensorflow.keras.initializers import RandomNormal
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
 
def MAPE(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = pd.read_csv (r'/Users/ewansaw/Desktop/BscProject/Models/Dataset-Full.csv')
df = df.dropna()

#define input sequence
loctime = np.array(df['Loctime'])
latitude = np.array(df['Latitude'])
alt = np.array(df['Altitude'])
e_density = np.array(df['Electron Density'])
e_temperature = np.array(df['Electron Temperature'])
density = np.array(df['Density'] * 0.25) #factor to account for efficiency 

#comment out 4 and 5 when generating maps
# convert to [rows, columns] structure
in_seq1 = loctime.reshape((len(loctime), 1))
in_seq2 = alt.reshape((len(alt), 1))
in_seq3 = latitude.reshape((len(latitude), 1))
#in_seq4 = e_density.reshape((len(e_density), 1))
#in_seq5 = e_temperature.reshape((len(e_temperature), 1))

out_seq = density.reshape((len(density), 1))
 
# horizontally stack columns
#dataset = hstack((in_seq1, in_seq2,  in_seq3, in_seq4, in_seq5, out_seq))
dataset = hstack((in_seq1, in_seq2,  in_seq3, out_seq))

# choose a number of time steps
n_steps = 1
# convert into input/output
X, y = split_sequences(dataset, n_steps) #could find more efficient way to input sequences
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

#creating shifted duplicated of each data set to train model on boundary condition
X_train1 = X_train.copy()
for i in range(len(X_train)):
    X_train1[i][0][0] = X_train[i][0][0] + 360
    
X_train2 = X_train.copy()
for i in range(len(X_train)):
    X_train2[i][0][0] = X_train[i][0][0] - 360
    
#updating new training set    
X_train = np.append(X_train, (X_train1, X_train2))
X_train = X_train.reshape((501, 1, 3)) #3 for 3 input sequences

#repeating y_train #boundary condition 
y_train = np.tile(y_train, 3)    
    

# defining functions (model etc)
#create model for finding error and optimising
def create_model(optimizer='rmsprop', activation = 'selu', loss = 'mae', nodes = 200, epochs = 500):
    # define model
    '''
    model = Sequential()     #Rayan & Ewan's Model
    model.add(Conv1D(nodes, kernel_size=1, activation=activation,input_shape=(n_steps, n_features)))
    model.add(LSTM(100, activation=activation))
    model.add(Dense(50, activation = 'selu'))
    model.add(Dense(25, activation = 'selu'))
    model.add(Dense(10, activation = 'selu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizer , loss = loss)
    model.fit(X_train, y_train, epochs=2000, verbose=1)
    

    model = Sequential()     #Rayan & Ewan's Model
    model.add(LSTM(nodes, activation = activation, input_shape=(n_steps, n_features)))
    model.add(Dense(100, activation = activation))
    model.add(Dense(50, activation = activation))
    model.add(Dense(25, activation = activation))
    model.add(Dense(10, activation = activation))
    model.add(Dense(1))
    model.compile(optimizer=optimizer , loss = loss )
    model.fit(X_train, y_train, epochs=1000, verbose=1, )
    '''
  # define model
    model = Sequential()     #Rayan & Ewan's Model
    model.add(LSTM(nodes, activation=activation, input_shape=(n_steps, n_features)))
    model.add(Dense(100, activation = activation))
    model.add(Dense(50, activation = activation))
    model.add(Dense(25, activation = activation))
    model.add(Dense(10, activation = activation))
    model.add(Dense(1))
    model.compile(optimizer=optimizer , loss = loss)
    model.summary()
    model.fit(X, y, batch_size = 10 , epochs=epochs, verbose=1)    

    
    # demonstrate prediction
    y_predicted = []
    for i in range(len(X_test)):
        x_input = X_test[i] 
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        #as there should not be any -ve values for density
        print(yhat)
        if yhat < 0:
            y_predicted.append(0)
        else:    
            y_predicted.append(yhat[0][0])
    
    plt.figure()
    plt.grid()
    plt.xlabel("Measurement Number")
    plt.ylabel("Density cm^-3")

    plt.plot(y_predicted, '.', color = "green")
    plt.plot(y_predicted, label = 'Predicted', color = "green")
    plt.plot(y_test, label = 'Recorded', color = 'orange')

    plt.plot(y_test, '.', color = "orange")
    plt.legend(loc = "upper left")
    plt.title("Ion Density")
    plt.show()

    rmse = sqrt(mean_squared_error(y_test, y_predicted)) #root mean square error 
    error = MAPE(y_test, y_predicted)
    return error, rmse

def find_error(n, n_nodes, epochs):
    mape = []
    rmse = []
    for i in range(n):
        error, rms = create_model(nodes = n_nodes, epochs = epochs)
        mape.append(error)
        rmse.append(rms)
    return ([np.mean(mape), np.std(mape)],[np.mean(rmse), np.std(rmse)])

#%% grid search
'''
KC = KerasClassifier(build_fn=create_model)
parameters = {​​​​'nodes' : [20,50,100],
    'activation' : ['relu','softmax','sigmoid','selu'],
          'optimizer':['adam','rmsprop', 'SGD'],
          'loss':['mse','mae']}​​​​
grid_search = GridSearchCV(estimator=KC , 
param_grid=parameters)
grid_search.fit(X_train,y_train)
 
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
 
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
'''
#%% finding errors
mape, rmse = find_error(1, 200, epochs = 1500)





