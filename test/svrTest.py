# Wanlin Cui

import csv
import datetime
import math
import numpy as np
import sys, os
import pickle
import time
from numpy import recfromcsv
import string

## Append the paths to your algorithms here.
sys.path.insert(1, os.path.join(sys.path[0], '../python/Tesla'));
sys.path.insert(1, os.path.join(sys.path[0], '../python/Svr'));
sys.path.insert(1, os.path.join(sys.path[0], '../python'));

## Import your algorithms here.
from Tesla import Tesla
from svr import SVR
from ContextEngineBase import Complexity

## For different tests, these values will vary.
inputFilePath = "SVRTestInput.csv"
outputFilePath = "SVRTestOutput.csv"
complexity = Complexity.secondOrder
numTrainingSamples = 96
numExecuteSamples = 96
inputFile = open(inputFilePath)
outputFile = open(outputFilePath)
inputReader = csv.reader(inputFile)
outputReader = csv.reader(outputFile)
csv = recfromcsv(inputFilePath, delimiter=',')
## Change the name of the algorithm to test it out.
algorithmTest = SVR(complexity, 1, 0, [0], {});
teslaTimestamps = {}
svrTimestamps = {}

totRow = 35040
numRow = 96
day_train_start = 0
day_train_end = 0
day_predict = 4

x_train = []
y_train = []
x_predict = []
x_real = []
y_real = []

preSum = 0
#read in csv and parse data to trainer

for i in range(numRow):
	j = i
	while j < totRow:
		row = csv[j]
		date = row[0]
		power = row[1]

		date = date.replace("/"," ")
		date = date.replace(":"," ")

		t = time.strptime(date, "%d %m %Y %H %M %S")
		tv = (t[3]*3600+t[4]*60+t[5])/(24*3600.0)
		preSum = preSum + power

		j = j + numRow
	avg = preSum / 365
	x_obs = [tv]
	y_obs = avg
	x_train.append(x_obs)
	y_train.append(y_obs)
	
	firstTS = time.time()
	algorithmTest.addSingleObservation(x_obs, y_obs)
	secondTS = time.time()
	svrTimestamps["load" + str(i-numRow*day_train_start)] = secondTS - firstTS
	preSum = 0

firstTS = time.time()
algorithmTest.train()
secondTS = time.time()
teslaTimestamps["train"] = secondTS - firstTS

runningTotal = 0
y_predict = np.empty([0])

for i in range(numRow*day_predict, numRow*(day_predict+1)):
	row = csv[i]
	date = row[0]
	power = row[1]
	date = date.replace("/"," ")
	date = date.replace(":"," ")
	t = time.strptime(date, "%d %m %Y %H %M %S")
	tv = (t[3]*3600+t[4]*60+t[5])/(24*3600.0)

	x_real.append([tv])
	y_real.append(power)

	x_predict = [i]
	firstTS = time.time()
	theor = algorithmTest.execute(x_predict)
	y_predict = np.concatenate((y_predict, theor))

	secondTS = time.time()
	svrTimestamps["test" + str(i-numRow*day_predict)] = secondTS - firstTS
	svrTimestamps["delta" + str(i-numRow*day_predict)] = abs(power - theor)
	runningTotal += power;

length = len(y_real)
a = 0
suma = 0

for i in range(len(y_real)):
	a = ( (y_real[i].astype(float)) - (y_predict[i].astype(float)) ) / (y_real[i].astype(float))
	suma = suma + a

runningMAE = 0.0
runningMAE = suma / len(y_real)


avgActual = runningTotal/(1.0*numExecuteSamples)
netLoadingTime = 0;
for i in range(numTrainingSamples):
    netLoadingTime += svrTimestamps["load" + str(i)]

print("Time (tot): " + str(netLoadingTime) + " seconds")
print("Time (avg): " + str(netLoadingTime/(1.0*numTrainingSamples)) + " seconds")
print("MAE: " + str(runningMAE))
