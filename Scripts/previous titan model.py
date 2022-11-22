
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
from sklearn.model_selection import train_test_split
import pandas as pd

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

df = pd.read_csv (r'/Users/ewansaw/Desktop/BscProject/Model/Dataset-Full.csv')
print (df)

df = df.dropna()

# define input sequences
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
#in_seq4 = e_density.reshape((len(e_density), 1))
#in_seq5 = e_temperature.reshape((len(e_temperature), 1))


out_seq = density.reshape((len(density), 1))

# horizontally stack columns
#dataset = hstack((in_seq1, in_seq2,  in_seq3, in_seq4, in_seq5, out_seq))
dataset = hstack((in_seq1, in_seq2,  in_seq3, out_seq))
#%%

# convert into input/output
X, y = split_sequences(dataset, n_steps) #could find more efficient way to input sequences

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# define model

model = Sequential()     #Rayan & Ewan's Model
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()


# fit model
#model.fit(X_train, y_train, epochs=500, verbose=1)
model.fit(X, y, epochs=500, verbose=1)

#%%
def contour_map(alt = 1300, n_levels = 50): #alt in km, n_levels is number of contour levels
    n_long = 361 #number of longtiude points (x-axis)
    n_lat = 181 #number of latitude points (y-axis)
    altitude = np.repeat(alt, n_lat * n_long) #generate all altitudes

    #create grid of points for all coordinates
    long = np.linspace(-180, 180, n_long)
    long = np.tile(long, n_lat) 
    lat = np.linspace(-90, 90, n_lat)
    lat = np.repeat(lat, n_long)
    heat_map = np.stack((long, altitude, lat), axis=1)
    X_map = heat_map.reshape((n_lat * n_long, 1, 3))

 
    #generate prediction
    y_predicted = []
    for i in range(n_lat * n_long):
       x_input = X_map[i] 
       x_input = x_input.reshape((1, n_steps, n_features))
       yhat = model.predict(x_input, verbose=0)
       y_predicted.append(yhat[0][0])   
    map_data = 0.25 * np.array(y_predicted) #factor of 0.25 to make up for efficiency of detector
 
 
 
    #plot contour
    plt.contour(long.reshape(n_long, n_lat), lat.reshape(n_long, n_lat), 
                map_data.reshape(n_long, n_lat), levels = n_levels, 
                antialiased = True)
    #explore plt.contour arguments to see if there are other ways to improve plot
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar()
    plt.title("Ion Density prediction map at altitude of %s km"%alt)
#%%
model = create_model()
contour_map()


    
'''
plan:
optimise model to make spatial maps, remember to train using 100% of data
check if function i made is correct to make plots,
----used 100 percent of data

check if sin parts work, see if there is a better way to do it 
-> use percentage error to check with method is better?? sine or no sine
----decided not to use sin

run model 5 times to create plots for same altitude, then take average
run this for different altitudes 
----done

make function to plot for sliced altitude, y altitude, x latitude 
at 0, -150 and +150 long
----done

remember actual density is *0.25 to account for efficiency (y)
-> results should be in range 0-10
->density should be higher at 1000km than 1300 km
-> no -ve values
----done 




remember to collect resources page for refereencing later in report

read papers 


stitch 3 datasets togetehr and train on them then test using 1 


figure out getting the quantitative values right 
use a systematic tes 

vary a node vary layers 


could plot the data itself over contuour map 

get the first step right the 80/20 fix

maybe you do wantt to have a fixed 80 and fixed 20 

another test pick a set up that you think is correct 


TMR:

ask about using imperial computers
fix sin thing 




 week: 23/11/20
    
    
    
question about direction of latitude 
question about locatl time and day and night side
see paper and code
    
https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
    
    
    

plot result percentage errorsto validate these choices and have sources to explain why

boundary value problem
only need to stitch data left and right  (padded)
see chat 

values for report 
    


why error instead of rms
easier to explain percenage error instead of rms?


oveerfitting and underfitting



figure out which error to use 


do different activation functions interact differently with differenet optimisers



Notes for 30/11:
    
    save plots as csv files 
    lat vs long with adjusted colour bar ranges 
    can run for 2500 epochs 
    find out dawn or dusk
    read paper 
    lit review for different times 
    how different optimsers interact with activation functions 
    different types of layers 

log data to produce different colour bar 

[15:45] Desai, Ravindra T
2. lat vs long with individual colorbar range

[15:59] Desai, Ravindra T
4. research into next step

[16:01] Desai, Ravindra T
5. can you produce 1 plot,  lat vs alt at 0 degrees with >2000 epochs, and 200 and 1000 layers. Produce log and non log plot


essentially try to strecth the data, see if u can catch out any anomalies because it is very smooth 


always gnna be liited by data but see if u can strecth to make new directions 
see if viable means for research


say this is a negtaively charged ion  compare to postivley charged ion 

compare tdifferent ions 


compare what flybys match up with data and check if csv file contains all data




Today (7/12):
    
what does conv1d layer  do

averaging smoothens out data so if we want to catch certain things gotta try not to 

plot log scale?

pool layer after conv1d

conv 1d/2d/3d???



2000 epochs 
20
200
1000


run for 
20, 100, 200, 500, 1000, 2000

200 - 2000 on log scale 


for contour maps 
 
 
 
 5 runs for each epochs 
 see variablility and average 
 5 plots and an average 
 do for alt and long plots (x2)
 
 
 1000 km  
 onvergence study for real data 
 
 
 final plots 
 use final setup for 4 long v lat plots 
 
 
 are we overtraining with this??
 how are you meant to find out ?
 
 
 
 
 noticed that as you increase epochs, get a bit more complexity, except there
 is a possibility of overtraining? 
 
