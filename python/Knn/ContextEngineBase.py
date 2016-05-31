from enum import Enum

import math
import numpy as np

class Complexity(Enum):
    firstOrder  = 1
    secondOrder = 2
    thirdOrder  = 3
    fourthOrder = 4
    fifthOrder  = 5
    


## Implementation of the context engine base class: the class inherited by other
## machine learning algorithms.
class ContextEngineBase:

    ## Member variables
    #  Function order - limit the highest order of the function
    complexity = Complexity.firstOrder;

    #  Number of inputs - interface for the number of input variables -
    #  defines input vector (+1 for training vector - n input, 1 output)
    numInputs = 0;

    #  Classification of the output - 0 is continuous, 1+ is # of states
    outputClassifier = 0;

    #  Classification of the inputs as an in-order list
    inputClassifiersList = [];

    #  Number of observations - a running count of the unique numbe of
    #  observations
    numObservations = 0;

    #  Additional custom algorithm-specific outputs as a key-value dictionary
    customFieldsDict = {};
    
    #  Matrix model - each row represents a new input vector
    observationMatrix = np.empty([0, 0]);

    #  Coefficient vector - the column vector representing the trained
    #  coefficients based on observations
    coefficientVector = [];

    #  Output observation vector - the column vector of recorded observations
    outputVector = [];
    

    #  Constructor - the order and number of inputs are mandatory
    #  Parameters:
    #    complexity: an instance of the Complexity enumerated type
    #    numInputs: integer number of inputs
    #    outputClassifier: integer for discrete (#) or continuous (0) output
    #    inputClassifiers: list of integers for discrete/continuous inputs
    #    appFieldsDict: dictionary of key/value pairs of app-specific fields
    def __init__(self,
                 complexity,
                 numInputs,
                 outputClassifier,
                 inputClassifiers,
                 appFieldsDict):

        if (len(inputClassifiers) != numInputs):
            raise ValueError("The magnitude of inputClassifiers",
                             "must be the same as numInputs");
        self.complexity = complexity;
        self.numInputs = numInputs;
        self.outputClassifier = outputClassifier;
        self.inputClassifiersList = inputClassifiers;
        self.customFieldsDict = appFieldsDict;

        # Generate the blank coefficient matrix
        self.coefficientVector = np.zeros([self.numInputs,1])

        # All other matrices/vectors are left the same, as they are dependent
        # on the number of observations.

    #  Add a new training observation. Requirements: newInputObs must be a
    #  row array of size numInputs. newOutputObs must be a single value.
    def addSingleObservation(self, newInputObs, newOutputObs):
        if (len(newInputObs) == self.numInputs
            and type(newOutputObs) not in (tuple, list)):

            # Only add non-duplicates
            if (not self.isADuplicate(newInputObs, newOutputObs)):
                # TODO: Replace the following code with a general implementation
                if (self.observationMatrix.shape[0] == 0):
                    self.observationMatrix = np.array([newInputObs]);
                    self.outputVector = np.array([newOutputObs]);
                    self.numObservations = 1;
                else:
                    self.observationMatrix = np.append(self.observationMatrix,\
                                                       np.array([newInputObs]),\
                                                       axis=0);
                    self.outputVector = np.append(self.outputVector,\
                                                  np.array([newOutputObs]),\
                                                  axis=0);
                    self.numObservations += 1;
        else:
            print("Wrong dimensions!");


    #  Add a set of training observations, with the newInputObsMatrix being a
    #  set of correctly-sized vectors and newOutputVector being a vector of
    #  individual values.
    def addBatchObservations(self, newInputObsMatrix, newOutputVector):
        for newInputVector in newInputObsMatrix:
            outputValue = newOutputVector.pop();
            self.addSingleObservation(newInputVector, outputValue);

    #  Train the coefficients on the existing observation matrix if there are
    #  enough observations.
    def train(self):
        if (self.observationMatrix.shape[0] >= self.numNormalizedInputs):
            print("Training started");
            self.coefficientVector = \
                np.linalg.lstsq(self.observationMatrix, self.outputVector);
        else:
            print("Not enough observations to train!");
    
    #  Returns True if the provided input vector and output observation already
    #  exist in the observation matrix, False otherwise
    def isADuplicate(self, inputVector, outputObs):
        for row in range(0, self.observationMatrix.shape[0]):
            if (np.array_equal(self.observationMatrix[row], inputVector) \
               and self.outputVector[row] == outputObs):
                return True;
        return False;

    #  Test the trained matrix against the given input observation
    def test(self, inputObsVector):
        return np.dot(self.coefficientVector[0],inputObsVector);
