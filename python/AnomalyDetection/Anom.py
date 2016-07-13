import numpy as np
import math
import scipy
import Anomaly
from ContextEngineBase import *



class Anom(ContextEngineBase):

    x_Obs = []
    y_Obs = []
    y_test = np.empty([0])
    Thresh = 0
        
    def __init__(self, complexity, numInputs, outputClassifier, inputClassifiers, appFieldsDict):
        ContextEngineBase.__init__(self,complexity, numInputs, outputClassifier, inputClassifiers, appFieldsDict)
        #self.discreteOutputs = discreteOutputs;
        #3sself.discreteInputs = discreteInputs;
        self.x_Obs = np.empty([0,numInputs]);
        self.x_Test = np.empty([0,numInputs]);                    #self.knnRegressor = KNeighborsRegressor(n_neighbors=self.complexity.value, weights='uniform');
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



    def train(self):
        if(self.numObservations > 0):
            print("Training Started");
            #Use module from Anomaly detection
            self.Thresh = Anomaly.AnomThresh(self.x_Obs);
            return True;
        else:
            print("Not enough observations to train");
            return False;

    def execute(self,inputObsVector):
        inc = 0;
        print len(inputObsVector)

        data = inputObsVector;

        while(inc < len(data)):
            if(data[inc] > self.Thresh):
                inputObsVector.remove(data[inc])

            inc = inc + 1;

                
        outputObsVector = data;
        return inputObsVector;



