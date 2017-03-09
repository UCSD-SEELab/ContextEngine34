#!/usr/bin/python

# CSC
# Input: 	Air quality, Geographical data
# Output: 	Asthma Hospitalization rates
# Use TESLA algorithm
# Training set rotation

import csv
import datetime
import math
import numpy as np
import sys, os
import pickle
import time
import subprocess

## Append the paths to your algorithms here.
sys.path.insert(1, os.path.join(sys.path[0], '../python/Tesla'));
sys.path.insert(1, os.path.join(sys.path[0], '../python'));

## Import your algorithms here.
from Tesla import Tesla
#from Knn import Knn
from ContextEngineBase import Complexity

## Set your options here.
inputFilePath = "aqTestInput.csv"
outputFilePath = "aqTestOutput.csv"
trainSetProportion = 0.80
complexityOptions = [ Complexity.firstOrder, Complexity.secondOrder, Complexity.thirdOrder ]
numTrials = 5
inputMask = [1] * 100 # the don't care option

cmd = "awk -F, '{print NF; exit}' " +inputFilePath
p = subprocess.Popen(cmd,stdout=subprocess.PIPE, shell=True).communicate()
numInputsAvailable = int(p[0].strip())

if len(inputMask) != numInputsAvailable:
	print("Input mask length does not match number of inputs provided, defaulting to all available inputs");
	inputMask = [1] * numInputsAvailable

numInputs = min(sum(inputMask), numInputsAvailable)

cmd = "wc -l < " +inputFilePath
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()
numTotalSamples = int(p[0].strip())

numTrainingSamples = int(numTotalSamples * trainSetProportion);
numExecuteSamples = numTotalSamples - numTrainingSamples;

print('Total samples: ',numTotalSamples, 'Training samples: ',numTrainingSamples, 'Test samples: ',numExecuteSamples)

for complexity in complexityOptions:
	print(complexity, " (numTrials: ",numTrials, ")", sep="")

	for trial in range(numTrials):
		inputFile = open(inputFilePath);
		outputFile = open(outputFilePath);
		inputReader = csv.reader(inputFile);
		outputReader = csv.reader(outputFile);

		## Change the name of the algorithm to test it out.
		algorithmTest = Tesla(complexity, numInputs, 0, [0] * numInputs, {});
		teslaTimestamps = {};
		teslaError = {};
		knnTimestamps = {};

		rowOffset = int(numTrainingSamples * trial % numTotalSamples);
		for i in range(rowOffset):
			try:
				inputRow = next(inputReader);
				outputRow = next(outputReader);
			except:
				inputFile.seek(0);
				outputFile.seek(0);
				inputRow = next(inputReader);
				outputRow = next(outputReader);
		

		## Load samples and TRAIN
		for trainingSample in range(numTrainingSamples):
			# Depending on which trial, offset the starting point for obtaining training samples
			try:
				inputRow = next(inputReader);
				outputRow = next(outputReader);
			except:
				inputFile.seek(0);
				outputFile.seek(0);
				inputRow = next(inputReader);
				outputRow = next(outputReader);
		
			if (len(inputRow) > 0):
				inputVec = [ float(x) for x in inputRow ]
				output = float(outputRow[0]);

				firstTS = time.time();
				algorithmTest.addSingleObservation(inputVec, output);
				secondTS = time.time();
				teslaTimestamps["load" + str(trainingSample)] = secondTS - firstTS;
			else:
				print("Error reading training samples");
				sys.exit();

		firstTS = time.time();
		algorithmTest.train();
		secondTS = time.time();
		teslaTimestamps["train"] = secondTS - firstTS;

		## Load samples and TEST
		runningTotal = 0;
		for executeSample in range(numExecuteSamples):
			try:
				inputRow = next(inputReader);
				outputRow = next(outputReader);
			except:
				inputFile.seek(0);
				outputFile.seek(0);
				inputRow = next(inputReader);
				outputRow = next(outputReader);
		
			if (len(inputRow) > 0):
				inputVec = [ float(x) for x in inputRow ]
				output = float(outputRow[0]);

				firstTS 	= time.time();
				theor 		= algorithmTest.execute(inputVec);
				secondTS 	= time.time();
				teslaTimestamps["test" + str(executeSample)] = secondTS - firstTS;
				teslaError["delta" + str(executeSample)] = abs(output - theor);
				runningTotal += output;
			else:
				print("Error reading test samples");
				sys.exit();

		netLoadingTime = 0;
		for i in range(numTrainingSamples):
			netLoadingTime += teslaTimestamps["load" + str(i)];

		netExecuteTime = 0;
		runningMAE = 0.0;
		for i in range(numExecuteSamples):
			netExecuteTime += teslaTimestamps["test" + str(i)];
			runningMAE += teslaError["delta" + str(i)];

		avgActual = runningTotal/(1.0*numExecuteSamples);
		runningMAE = runningMAE/(1.0*avgActual*numExecuteSamples);

		#print("Loading time (tot): %.6f seconds" % netLoadingTime );
		#print("Loading time (avg): %.6f seconds" % (netLoadingTime/(1.0*numTrainingSamples)) );
		#print("Training time: %.6f seconds" % teslaTimestamps["train"] );
		#print("Execute time (tot): %.6f seconds" % netExecuteTime );
		#print("Execute time (avg): %.6f seconds" % (netLoadingTime/(1.0*numExecuteSamples)) );
		print("MAE: %.6f" % runningMAE );
		#print("%.6f" % runningMAE );

		inputFile.close()
		outputFile.close()

