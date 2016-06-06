#KnnTest 
#Xinyuan Zhang

import csv
import datetime
import math
import numpy as np
import sys, os
import pickle
import time
import string
from numpy import recfromcsv
## Append the paths to your algorithms here.
sys.path.insert(1, os.path.join(sys.path[0], '../python/Tesla'));
sys.path.insert(1, os.path.join(sys.path[0], '../python/Knn'));

## Import your algorithms here.
from Tesla import Tesla
from Knn import Knn
from ContextEngineBase import Complexity

## For different tests, these values will vary.
inputFilePath = "dish.csv"
outputFilePath = "dishOutput.csv"
inputFile = open(inputFilePath);
outputFile = open(outputFilePath);
inputReader = csv.reader(inputFile);
outputReader = csv.reader(outputFile);
complexity = Complexity.secondOrder;
numTrainingSamples = 96;
numExecuteSamples = 96;
inputReader = csv.reader(inputFile);
outputReader = csv.reader(outputFile);

csv = recfromcsv(inputFilePath, delimiter=',')
## Change the name of the algorithm to test it out.
algorithmTest = Knn(Complexity.secondOrder, 7, 0, [0,0,0,0,0,0,0], {});
teslaTimestamps = {};
knnTimestamps = {};


numRow = 96
day_train_start=0
day_train_end=0
day_predict_start= 0
day_predict_end = 0
#read in csv and parse data to trainer

for i in range(numRow*day_train_start,numRow*(day_train_end+1)):
	row = csv[i]
	date=row[0]
	date=date.decode()
	dishwasher=csv[i+1][3]
	date=date.replace("/" , " ")
	date=date.replace(":" , " ")
	t=time.strptime(date, "%m %d %Y %H %M")
	ti = (t[3]*3600+t[4]*60+t[5])/(24*3600.0)
	x_obs = [ti, row[2], row[4], row[5], row[6], row[7], row[8]]
	y_obs = dishwasher
	firstTS = time.time();
	algorithmTest.addSingleObservation(x_obs, y_obs);
	secondTS = time.time();
	knnTimestamps["load" + str(i-numRow*day_train_start)] = secondTS - firstTS;

firstTS = time.time();
algorithmTest.train();
secondTS = time.time();
knnTimestamps["train"] = secondTS - firstTS;

runningTotal = 0;

for i in range(numRow*day_predict_start,numRow*(day_predict_end+1)):
	row = csv[i]
	date=row[0]
	date=date.decode()
	date_predict = csv[i+1][0]
	output=round(csv[i+1][3],4)
	date=date.replace("/" , " ")
	date=date.replace(":" , " ")
	t=time.strptime(date, "%m %d %Y %H %M")
	ti = (t[3]*3600+t[4]*60+t[5])/(24*3600.0)
	x_predict=[ti, row[2], row[4], row[5], row[6], row[7], row[8]];
	firstTS = time.time();
	theor = algorithmTest.execute(x_predict);
	secondTS = time.time();
	knnTimestamps["test" + str(i-numRow*day_predict_start)] = secondTS - firstTS;
	knnTimestamps["delta" + str(i-numRow*day_predict_start)] = abs(output - theor);
	runningTotal += output;

avgActual = runningTotal/(1.0*numExecuteSamples);
netLoadingTime = 0;
for i in range(numTrainingSamples):
    netLoadingTime += knnTimestamps["load" + str(i)];

netExecuteTime = 0;
runningMAE = 0.0;
for i in range(numExecuteSamples):
    netExecuteTime += knnTimestamps["test" + str(i)];
    runningMAE += knnTimestamps["delta" + str(i)];

runningMAE = runningMAE/(1.0*avgActual*numExecuteSamples);

print("Loading time (tot): " + str(netLoadingTime) + " seconds");
print("Loading time (avg): " + str(netLoadingTime/(1.0*numTrainingSamples)) + " seconds");
print("Training time: " + str(knnTimestamps["train"]) + " seconds");
print("Execute time (tot): " + str(netExecuteTime) + " seconds");
print("Execute time (avg): " + str(netLoadingTime/(1.0*numExecuteSamples)) + " seconds");
print("MAE: " + str(runningMAE));