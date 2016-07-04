import numpy as np
import sys, os
from sklearn.cluster import KMeans
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ContextEngineBase import *

class Kmeans(ContextEngineBase):


	x_Obs = []

	# Output observation array
	# eg. y = [0, 1]
	km = None

	def __init__(self, complexity, numInputs, outputClassifier, inputClassifiers, appFieldsDict):
		ContextEngineBase.__init__(self, complexity, numInputs, outputClassifier, inputClassifiers, appFieldsDict)
		self.km = KMeans(n_clusters= self.complexity, random_state=0)
		self.x_Obs = np.empty([0,self.numInputs]);


	#  Add a new training observation. Requirements: newInputObs must be a
    #  row array of size numInputs. newOutputObs is none.
	def addSingleObservation(self, newInputObs, newOutputObs=None): 
		if (len(newInputObs) == self.numInputs):
			print("All good!");
			self.x_Obs = np.vstack((self.x_Obs,newInputObs));
			self.numObservations += 1;
		else:
			print("Wrong dimensions!");

	#  Add a set of training observations, with the newInputObsMatrix being a
	#  matrix of doubles, where the row magnitude must match the number of inputs,
	#  and the column magnitude must match the number of observations.
	#  and newOutputVector is none
	def addBatchObservations(self, newInputObsMatrix, newOutputVector=None):

		if(len(newInputObsMatrix.shape) == 2 and newInputObsMatrix.shape[1] == self.numInputs):
			print("All good!");
			for newInputVector in newInputObsMatrix:
				self.addSingleObservation(newInputObs=newInputVector);
		else:
			print("Wrong dimensions!");

	#  Train the coefficients on the existing observation matrix if there are
    #  enough observations.
	def train(self):
		if (self.numObservations > 0):
			print("Training started");
			self.km.fit(self.x_Obs);
			return True;
		else:
			print("Not enough observations to train!");
			return False;


	#  Execute the trained matrix against the given input observation
    #	inputObsVector is a row vector of doubles
    # return the center of the inputObsVector
	def execute(self, inputObsVector=None):
		if(len(inputObsVector) == self.numInputs):
			print("Begin execute");
			x_Sample = np.reshape(inputObsVector,(1,self.numInputs));
			x_Res = self.km.predict(x_Sample);
			index = x_Res[0]
			return km.cluster_centers_[index]
		else:
			print("Wrong dimensions, fail to execute");
			return None;
