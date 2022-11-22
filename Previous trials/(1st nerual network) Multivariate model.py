# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:18:48 2020

@author: rayan
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from numpy import array
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
#%%

# choose a number of time steps
n_steps = 1


# define input sequence
in_seq1 = np.array([])
for i in range(1000):
    in_seq1 = np.append(in_seq1,np.random.randint(0,50)) 

in_seq2 = np.array([])
for i in range(1000):
    in_seq2 = np.append(in_seq2,np.random.randint(0,50))
#in_seq1 = array([5*(X[i]**2) for i in range(len(X))])


out_seq = array([(in_seq1[i]**2)+3*(in_seq2[i]**1) for i in range(len(in_seq1))])
# convert to [rows, columns] structure

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

# convert into input/output
X, y = split_sequences(dataset, n_steps)

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=200, verbose=1)
#%%
# demonstrate prediction
x_input = array([6, 2])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
