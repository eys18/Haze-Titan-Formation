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
from tensorflow.keras.layers import Dense, Conv1D
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
altitude = np.array(df['Altitude'])
e_density = np.array(df['Electron Density'])
e_temperature = np.array(df['Electron Temperature'])
density = np.array(df['Density'] * 0.25) #factor to account for efficiency 

#comment out 4 and 5 when generating maps
# convert to [rows, columns] structure
in_seq1 = loctime.reshape((len(loctime), 1))
in_seq2 = altitude.reshape((len(altitude), 1))
in_seq3 = latitude.reshape((len(latitude), 1))
out_seq = density.reshape((len(density), 1))
 
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2,  in_seq3, out_seq))

# choose a number of time steps
n_steps = 1
# convert into input/output
X, y = split_sequences(dataset, n_steps) #could find more efficient way to input sequences

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

#creating shifted duplicated of each data set to train model on boundary condition
X1 = X.copy()
for i in range(len(X)):
    X1[i][0][0] = X[i][0][0] + 360
    
X2 = X.copy()
for i in range(len(X)):
    X2[i][0][0] = X[i][0][0] - 360

#updating new training set by stitching them together
X = np.append(X, (X1, X2))
X = X.reshape((627, 1, 3)) #3 for 3 input sequences

#repeating y
y = np.tile(y, 3)    
    

# defining functions (model etc)
#create model for contour map
def create_model(optimizer='rmsprop', activation = 'selu', loss = 'mae', nodes = 200, epochs = 200):
    # define model
    model = Sequential()     #Rayan & Ewan's Model
    model.add(LSTM(nodes, activation=activation, input_shape=(n_steps, n_features)))
    model.add(Dense(100, activation = activation))
    model.add(Dense(50, activation = activation))
    model.add(Dense(25, activation = activation))
    model.add(Dense(10, activation = activation))
    model.add(Dense(1))
    model.compile(optimizer=optimizer , loss = loss )
    model.fit(X, y, batch_size = 10, epochs=epochs, verbose=1)    
    return model

    '''
    model = Sequential()     #Rayan & Ewan's Model
    model.add(Conv1D(nodes, kernel_size=1, activation=activation,input_shape=(n_steps, n_features)))
    model.add(LSTM(100, activation=activation))
    model.add(Dense(50, activation = activation))
    model.add(Dense(25, activation = activation))
    model.add(Dense(10, activation = activation))
    model.add(Dense(1))
    model.compile(optimizer=optimizer , loss = loss)
    model.fit(X, y, epochs=200, verbose=1)
    return model
    '''
    
def find_error(n, n_nodes):
    mape = []
    rmse = []
    for i in range(n):
        error, rms = create_model(nodes = n_nodes)
        mape.append(error)
        rmse.append(rms)
    return ([np.mean(mape), np.std(mape)],[np.mean(rmse), np.std(rmse)])

def contour_map(alt = 1300):#, n_levels = 50): #alt in km, n_levels is number of contour levels
    n_long = 361 #number of longtiude points (x-axis)
    n_lat = 181 #number of latitude points (y-axis)
    altitude = np.repeat(alt, n_lat * n_long) #generate all altitudes

    #create grid of points for all coordinates
    long = np.linspace(-180, 180, n_long) #x-axis
    long = np.tile(long, n_lat) 
    lat = np.linspace(-90, 90, n_lat) #y-axis
    lat = np.repeat(lat, n_long)
    heat_map = np.stack((long, altitude, lat), axis=1)
    X_map = heat_map.reshape((n_lat * n_long, 1, 3))

    #generate prediction
    y_predicted = []
    for i in range(n_lat * n_long):
       x_input = X_map[i] 
       x_input = x_input.reshape((1, n_steps, n_features))
       yhat = model.predict(x_input, verbose=0)
       
       #as there should not be any -ve values for density
       if yhat < 0:
           y_predicted.append(0)
       else:    
           y_predicted.append(yhat[0][0])
    map_data = np.array(y_predicted) 
 
    return  (long.reshape(n_long, n_lat), lat.reshape(n_long, n_lat), 
             map_data.reshape(n_long, n_lat))

#slicing prediction, with x-axis latitude and y-axis altitude
def contour_map_slice(long = 0):#, n_levels = 50): #alt in km, n_levels is number of contour levels
    n_lat = 181 #number of latitude points (x-axis)
    n_alt = 161 #number of altitude points (y-axis) chosen to give increments of 2.5 km
    
    longitude = np.repeat(long, n_lat * n_alt) #generate all longitudes

    #create grid of points for all coordinates
    lat = np.linspace(-90, 90, n_lat) #x-axis
    lat = np.tile(lat, n_alt)
    alt = np.linspace(950, 1350, n_alt) #y-axis
    alt = np.repeat(alt, n_lat) 
    heat_map = np.stack((longitude, alt, lat), axis=1)
    X_map = heat_map.reshape((n_lat * n_alt, 1, 3))

    #generate prediction
    y_predicted = []
    for i in range(n_lat * n_alt):
       x_input = X_map[i] 
       x_input = x_input.reshape((1, n_steps, n_features))
       yhat = model.predict(x_input, verbose=0)
       
       #as there should not be any -ve values for density
       if yhat[0][0] < 0:
           yhat[0][0] = 0
       y_predicted.append(yhat[0][0])   
    map_data = np.array(y_predicted) 
    
    return  (lat.reshape(n_lat, n_alt), alt.reshape(n_lat, n_alt), 
             map_data.reshape(n_lat, n_alt))



#%% generating contour
#set altitude in km
alt_choice = 1300
#number of cycles
n = 5
epochs = 1500
# generating model and conotour map 5 times and plotting the average
map_data = np.zeros((n, 361, 181))

for i in range(n):
    model = create_model(epochs = epochs)
    long, lat, maps = contour_map(alt = alt_choice)
    plt.contour(long, lat, maps, levels = 500, antialiased = True)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar()
    plt.title("Ion Density prediction map at altitude of %s km"%alt_choice)
    plt.show()
    map_data[i] = maps
#now average the data by dividing by n
map_datas = np.sum(map_data, axis=0) / n
#log_map = np.log(map_data)
#plotting contour
levels = np.linspace(0,9,300)
plt.contour(long, lat, map_datas, levels = levels, antialiased = True)
#plt.contour(long, lat, log_map, levels = levels, antialiased = True)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar()
plt.title("Ion Density prediction map at altitude of %s km"%alt_choice)
plt.show()

# saving data to .csv/.dat file
output = np.column_stack((long.flatten(),lat.flatten(), map_datas.flatten()))
#note: shapes of each columnn of .csv are originially  (361,181) to plot contour

np.savetxt('%skm_alt(%s epochs).csv'%(alt_choice, epochs), output, delimiter=',')
 
#%% plotting sliced contour
#set longitude
long_choice = 0
n = 5
epochs = 1500
# generating model and conotour map n times and plotting the average
map_data = np.zeros((n, 181, 161))
    
for i in range(n):
    model = create_model(epochs = epochs)
    lat, alt, maps = contour_map_slice(long = long_choice)
    plt.contour(lat, alt, maps, levels = 501, antialiased = True)
    plt.xlabel("Latitude")
    plt.ylabel("Altitude")
    plt.colorbar()
    plt.title("Ion Density prediction map at Longitude of %s degrees"%long_choice)
    plt.show()
    map_data[i] = maps
#now average the data by dividing by n
map_datas = np.sum(map_data, axis=0) / n
levels = np.linspace(0, 7.36, 501)
plt.contour(lat, alt, map_datas, levels = 501, antialiased = True)
#plt.contour(lat, alt, log_map, levels = levels, antialiased = True)

plt.xlabel("Latitude")
plt.ylabel("Altitude")
plt.colorbar()
plt.title("Ion Density prediction map at Longitude of %s degrees"%long_choice)
plt.show()

#%%
# saving data to .csv/.dat file
output = np.column_stack((lat.flatten(), alt.flatten(), map_datas.flatten()))
#note: shapes of each columnn of .csv are originially (181, 161) to plot contour
np.savetxt('%sdeg_long(%s epochs).csv'%(long_choice, epochs), output, delimiter=',')
 