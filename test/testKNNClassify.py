#KNNClassifier 
#Dhanesh Pradhan

import csv
import datetime
import math
import numpy as np
import sys, os
import pickle
import time
from numpy import recfromcsv


## Append the paths to your algorithms here.
sys.path.insert(1, os.path.join(sys.path[0], '../python/KNNClassify'));
sys.path.insert(1, os.path.join(sys.path[0], '../python'));

## Import your algorithms here.
from Tesla import Tesla
from KNNClassify import KNNClassify
from ContextEngineBase import Complexity

## For different tests, these values will vary.
inputFilePath = "KNNClassifierInput.csv"
outputFilePath = "KNNClassifierOutput.csv"
inputFile = open(inputFilePath,'r');
outputFile = open(outputFilePath,'r');
inputReader = csv.reader(inputFile);
outputReader = csv.reader(outputFile);
complexity = Complexity.secondOrder;
numTrainingSamples = 60;
numExecuteSamples = 20;


## Change the name of the algorithm to test it out.
KNN = KNNClassify (Complexity.secondOrder, 11, 0, [0,0,0,0,0,0,0,0,0,0,0], {})
KNNClassifyTimestamps = {};



#read in csv and parse data to trainer
for trainingSample in range(numTrainingSamples):
    inputrow = next(inputReader);
    outputrow = next(outputReader);
    if (len(inputrow) > 10):
        inputs = np.asarray(inputrow[0:11], dtype=np.float32)
        outputs = np.asarray(outputrow[0], dtype=np.float32)
        firstTS = time.time();
        KNN.addSingleObservation(inputs,outputs);
        secondTS = time.time();
        KNNClassifyTimestamps["load" + str(trainingSample)] = secondTS - firstTS;

firstTS = time.time()
KNN.train();
secondTS = time.time();
KNNClassifyTimestamps["train"] = secondTS - firstTS;

runningTotal = 0;

for executeSample in range(numExecuteSamples):
    inputrow = next(inputReader);
    outputrow = next(outputReader);
    if (len(inputrow) > 10):
        inputs = np.asarray(inputrow[0:11], dtype=np.float32)
        outputs = np.asarray(outputrow[0], dtype=np.float32)
        firstTS = time.time();
        theor=KNN.execute(inputs)[0].item()
        secondTS = time.time();
        KNNClassifyTimestamps["test" + str(executeSample)] = secondTS - firstTS;
        KNNClassifyTimestamps["delta" + str(executeSample)] = abs(outputs - theor);
        runningTotal += outputs;
avgActual = runningTotal/(1.0*numExecuteSamples);

netLoadingTime = 0;

for i in range(numTrainingSamples):
    netLoadingTime += KNNClassifyTimestamps["load" + str(i)];

netExecuteTime = 0;
runningMAE = 0.0;
for i in range(numExecuteSamples):
    netExecuteTime += KNNClassifyTimestamps["test" + str(i)];
    runningMAE += KNNClassifyTimestamps["delta" + str(i)];

runningMAE = runningMAE/(1.0*avgActual*numExecuteSamples);

print("Loading time (tot): " + str(netLoadingTime) + " seconds");
print("Loading time (avg): " + str(netLoadingTime/(1.0*numTrainingSamples)) + " seconds");
print("Training time: " + str(KNNClassifyTimestamps["train"]) + " seconds");
print("Execute time (tot): " + str(netExecuteTime) + " seconds");
print("Execute time (avg): " + str(netLoadingTime/(1.0*numExecuteSamples)) + " seconds");
print("MAE: " + str(runningMAE));
