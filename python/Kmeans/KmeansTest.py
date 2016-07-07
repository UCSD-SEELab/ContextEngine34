#KmeansTest 
#Xinyuan Zhang

import csv
import datetime
import math
import numpy as np
import sys, os
import pickle
import time
from numpy import recfromcsv
import matplotlib.pyplot as plt


## Append the paths to your algorithms here.
sys.path.insert(1, os.path.join(sys.path[0], '../python/Tesla'));
sys.path.insert(1, os.path.join(sys.path[0], '../python/Kmeans'));
sys.path.insert(1, os.path.join(sys.path[0], '../python'));

## Import your algorithms here.
#from Tesla import Tesla
from Kmeans import Kmeans
from ContextEngineBase import Complexity

## For different tests, these values will vary.
inputFilePath = "refridge.csv"
#outputFilePath = "dishOutput.csv"
complexity = Complexity.secondOrder;

inputFile = open(inputFilePath);
#outputFile = open(outputFilePath);
inputReader = csv.reader(inputFile);
#outputReader = csv.reader(outputFile);
csv = recfromcsv(inputFilePath, delimiter=',')

algorithmTest = Kmeans(complexity, 96, 0, np.empty([96]), {});
teslaTimestamps = {};
kmeansTimestamps = {};

numRow = 96
day_train_start=0
day_train_end=29
#day_predict_start= 1
#day_predict_end = 1
#read in csv and parse data to trainer
numTrainingSamples = day_train_end-day_train_start+1;
#numExecuteSamples = 96;

x_obs = np.empty([0,96]);
for i in range(day_train_start,(day_train_end+1)):
	j=0
	dailyLoad=[]
	firstTS = time.time();
	#count for 96 rows, each 96 rows of data represents load curve of one day
	while (j<numRow): 
		row = csv[i*numRow+j]
		power=row[1]
		"""
		date=row[0]
		
		date=date.replace("/"," ")
		date=date.replace(":"," ")
		t=strptime(date, "%m %d %Y %H %M")
		time = (t[3]*3600+t[4]*60+t[5])/(24*3600.0)
		"""

		dailyLoad = np.append(dailyLoad, power)
		j = j+1
	
	algorithmTest.addSingleObservation(dailyLoad);
	x_obs = np.vstack((x_obs,dailyLoad))
	secondTS = time.time();
	kmeansTimestamps["load" + str(i)] = secondTS - firstTS;

firstTS = time.time();
algorithmTest.train();
secondTS = time.time();
kmeansTimestamps["train"] = secondTS - firstTS;

netLoadingTime = 0;
for i in range(numTrainingSamples):
    netLoadingTime += kmeansTimestamps["load" + str(i)];
print("Loading time (tot): " + str(netLoadingTime) + " seconds");
print("Loading time (avg): " + str(netLoadingTime/(1.0*numTrainingSamples)) + " seconds");
print("Training time: " + str(kmeansTimestamps["train"]) + " seconds");

centers = algorithmTest.km.cluster_centers_
T = np.linspace(0, 1,97)
T = np.delete(T,-1)
plt.subplot(1, 1, 1)

colors = ['r', 'b', 'k', 'm', 'c', 'y']
for obs in x_obs:
	plt.plot(T, obs, c='g')
for i in range(0,len(centers)):
	plt.plot(T, centers[i], c=colors[i], label='center'+str(i))
plt.axis('tight')
plt.legend()
plt.title("KMeans (#centers = %i, #days = %i)" % (complexity,numTrainingSamples))
plt.show()




