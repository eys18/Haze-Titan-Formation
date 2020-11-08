
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:18:48 2020

@author: rayan
"""
import numpy as np
import sklearn as sk
from numpy import array
import matplotlib.pyplot as plt
from numpy import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

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

# choose a number of time steps
n_steps = 1

#%%
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv (r'/Users/ewansaw/Desktop/Model/Dataset-Full.csv')
print (df)

df = df.dropna()
#%%

# define input sequence
loctime = np.array(df['Loctime'])
alt = np.array(df['Altitude'])
latitude = np.array(df['Latitude'])
e_density = np.array(df['Electron Density'])
e_temperature = np.array(df['Electron Temperature'])




density = np.array(df['Density'])
# convert to [rows, columns] structure
in_seq1 = loctime.reshape((len(loctime), 1))
in_seq2 = alt.reshape((len(alt), 1))
in_seq3 = latitude.reshape((len(latitude), 1))
in_seq4 = e_density.reshape((len(e_density), 1))
in_seq5 = e_temperature.reshape((len(e_temperature), 1))





out_seq = density.reshape((len(density), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2,  in_seq3, in_seq4, in_seq5, out_seq))
#%%

# convert into input/output
X, y = split_sequences(dataset, n_steps) #could find more efficient way to input sequences

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# define model

model = Sequential()     #Rayan & Ewan's Model
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

'''


model = Sequential() #George's Model'
model.add(Dense(80, input_shape=(n_steps, n_features), activation='selu'))
model.add(Dense(80, activation='selu'))
model.add(Dense(80, activation='selu'))
model.add(Dense(80, activation='selu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer="adam")
    #Display model\n",
model.summary()
'''
# fit model
model.fit(X_train, y_train, epochs=500, verbose=1)


#%%
# demonstrate prediction
y_predicted = []
for i in range(len(X_test)):
    x_input = X_test[i] 
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)
    y_predicted.append(yhat[0][0])


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_predicted)) #root mean square error 

our_data = y_predicted

#%%
plt.grid()
plt.xlabel("Measurement Number")
plt.ylabel("Density cm^-3")

plt.plot(our_data, '.', color = "green")
plt.plot(our_data, label = 'Predicted', color = "green")
plt.plot(y_test, label = 'Recorded', color = 'orange')
#plt.plot(georgedata, label = "Predicted")
plt.plot(y_test, '.', color = "orange")
plt.legend(loc = "upper left")
plt.title("Ion Density Prediction using George's Model")
