# Wanlin Cui

import numpy as np
import math
from sklearn import svm
from ContextEngineBase import ContextEngineBase
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

## Implementation of the SVR algorithm:
class SVR(ContextEngineBase):
	svrLinear = None
	x_Obs = []
	y_Obs = []
	y_Test = np.empty([0])
	
	def __init__(self, complexity, numInputs, outputClassifier, inputClassifiers, appFieldsDict):
		ContextEngineBase.__init__(self,complexity, numInputs, outputClassifier, inputClassifiers, appFieldsDict)
		self.x_Obs = np.empty([0, numInputs])
		self.x_Test = np.empty([0, numInputs])
		self.svrLinear = svm.SVR(kernel='rbf')
	
	def addSingleObservation(self, newInputObs, newOutputObs):
		if (len(newInputObs) == self.numInputs and type(newOutputObs) not in (tuple, list)):
			#print("All good!")
			self.x_Obs = np.vstack((self.x_Obs,newInputObs))
			self.y_Obs = np.append(self.y_Obs, newOutputObs)
			self.numObservations += 1
		else:
			print("Wrong dimensions!")
	
	def addBatchObservations(self, newInputObsMatrix, newOutputVector):
		if(len(newInputObsMatrix.shape) == 2 and newInputObsMatrix.shape[1] == self.numInputs and newOutputVector.shape[0] == newInputObsMatrix.shape[0]):
			#print("All good!");
			newOutputVector = newOutputVector.ravel();
			i = 0;
			for newInputVector in newInputObsMatrix:
				newOutputValue = newOutputVector[i];
				self.addSingleObservation(newInputVector, newOutputValue);
				i += 1;
		else:
			print("Wrong dimensions!");
		
	def train(self):
		if (self.numObservations > 0):
			#print("Training started");
			self.svrLinear.fit(self.x_Obs, self.y_Obs);
			return True;
		else:
			print("Not enough observations to train!");
			return False;
			
	def execute(self, inputObsVector):
		if(len(inputObsVector) == self.numInputs):
			#print("Begin execute");
    			#x_Test = np.vstack((self.x_Test,inputObsVector));
			x_Test = np.reshape(inputObsVector,(1,self.numInputs));
			self.y_Test = self.svrLinear.predict(x_Test);
			return self.y_Test;
		else:
			print("Wrong dimensions, fail to execute");
			return None;
