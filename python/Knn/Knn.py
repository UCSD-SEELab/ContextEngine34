import sys, os
import numpy as np
import math
sys.path.insert(1, os.path.join(sys.path[0], '..'));
from ContextEngineBase import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
## Implementation of the Knn algorithm: 
class Knn(ContextEngineBase):


	#  Number of observations - a running count of the unique numbe of
    #  observations
	#numObservations = 0;

    # Matrix model - each row represents a new input vector
    #eg. x_Obs = array([[ 1.,  2.,  3.],
    #   				[ 2.,  1.,  5.]])
	x_Obs = []

	# Output observation array
	# eg. y = [0, 1]
	y_Obs = []

	#x_Test = np.empty([0,0]);
	y_Test = np.empty([0]);

	#trained result
	knnRegressor = None;
	#knnClassifier = KNeighborsClassifier(n_neighbors=5);

	def __init__(self, complexity, numInputs, outputClassifier, inputClassifiers, appFieldsDict):
		ContextEngineBase.__init__(self,complexity, numInputs, outputClassifier, inputClassifiers, appFieldsDict)
		#self.discreteOutputs = discreteOutputs;
		#3sself.discreteInputs = discreteInputs;
		self.x_Obs = np.empty([0,numInputs]);
		self.x_Test = np.empty([0,numInputs]);
		self.knnRegressor = KNeighborsRegressor(n_neighbors=self.complexity.value, weights='uniform');
	#  Add a new training observation. Requirements: newInputObs must be a
    #  row array of size numInputs. newOutputObs must be a single value.
	def addSingleObservation(self, newInputObs, newOutputObs): 
		if (len(newInputObs) == self.numInputs and type(newOutputObs) not in (tuple, list)):
			print("All good!");
			self.x_Obs = np.vstack((self.x_Obs,newInputObs));
			#self.y_Obs = np.append(self.y_Obs, newOutputObs);
			#self.x_Obs = np.append(self.x_Obs, newInputObs);
			self.y_Obs = np.append(self.y_Obs, newOutputObs);
			self.numObservations += 1;
		else:
			print("Wrong dimensions!");

	#  Add a set of training observations, with the newInputObsMatrix being a
	#  matrix of doubles, where the row magnitude must match the number of inputs,
	#  and the column magnitude must match the number of observations.
	#  and newOutputVector being a column vector of doubles
	def addBatchObservations(self, newInputObsMatrix, newOutputVector):

		if(len(newInputObsMatrix.shape) == 2 and newInputObsMatrix.shape[1] == self.numInputs
			and newOutputVector.shape[0] == newInputObsMatrix.shape[0]):
			print("All good!");
			newOutputVector = newOutputVector.ravel();
			i = 0;
			for newInputVector in newInputObsMatrix:
				newOutputValue = newOutputVector[i];
				self.addSingleObservation(newInputVector, newOutputValue);
				i += 1;
		else:
			print("Wrong dimensions!");



    #  Train the coefficients on the existing observation matrix if there are
    #  enough observations.
	def train(self):
		if (self.numObservations > 0):
			print("Training started");
			self.knnRegressor.fit(self.x_Obs, self.y_Obs);
			return True;
		else:
			print("Not enough observations to train!");
			return False;

    #  Execute the trained matrix against the given input observation
    #	inputObsVector is a row vector of doubles
	def execute(self, inputObsVector):
		if(len(inputObsVector) == self.numInputs):
			print("Begin execute");
    		#x_Test = np.vstack((self.x_Test,inputObsVector));
			x_Test = np.reshape(inputObsVector,(1,self.numInputs));
			self.y_Test = self.knnRegressor.predict(x_Test);
			return self.y_Test[0];
		else:
			print("Wrong dimensions, fail to execute");
			return None;

