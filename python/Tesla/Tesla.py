import sys, os
import numpy as np
import math
sys.path.insert(1, os.path.join(sys.path[0], '..'));
                
from ContextEngineBase import *


## Implementation of the TESLA algorithm: map input variables to output
## observations
class Tesla(ContextEngineBase):

    ## Member variables
    #  Function order - limit the highest order of the function
    functionOrder = 1;

##    #  Number of inputs - interface for the number of input variables -
##    #  defines input vector (+1 for training vector - n input, 1 output)
##    numInputs = 0;
##
    #  Number of normalized inputs - this is the number of inputs needed,
    #  corrected for the order.
    #  e.g. 1st order == numInputs, 2nd order == numInputs^2, etc.
    numNormalizedInputs = 0;

##    #  Number of observations - a running count of the unique numbe of
##    #  observations
##    numObservations = 0;
##
##    #  Matrix model - each row represents a new input vector
##    observationMatrix = np.empty([0, 0]);
##
##    #  Coefficient vector - the column vector representing the trained
##    #  coefficients based on observations
##    coefficientVector = [];
##
##    #  Output observation vector - the column vector of recorded observations
##    outputVector = [];

    # History states - add x additional states of input and output history
    inputHistoryNum = 0;
    inputHistory = [];
    outputHistoryNum = 0;
    outputHistory = [];

    # Tracking values for each coefficient for correlation analysis
    correlationValues = {};
    

    #  Constructor - the order and number of inputs are mandatory
    def __init__(self,
                 complexity,
                 numInputs,
                 outputClassifier,
                 inputClassifiers,
                 appFieldsDict):
        super(Tesla, self).__init__(complexity,
                     numInputs,
                     outputClassifier,
                     inputClassifiers,
                     appFieldsDict);
        
        self.functionOrder = complexity.value;
        if ("inputHistoryNum" in self.customFieldsDict):
            self.inputHistoryNum = int(
                self.customFieldsDict["inputHistoryNum"]);
        if ("outputHistoryNum" in self.customFieldsDict):
            self.outputHistoryNum = int(
                self.customFieldsDict["outputHistoryNum"]);

        # Generate the number of normalized inputs. This is compounded by
        # the input and output histories
        self.numNormalizedInputs = \
            int(math.factorial(self.numInputs+self.functionOrder)/\
            (math.factorial(self.functionOrder)*math.factorial(self.numInputs)));

        # Update with input history
        self.numNormalizedInputs += (self.numNormalizedInputs
                                     * self.inputHistoryNum);

        # Update with output history
        self.numNormalizedInputs += self.outputHistoryNum;

        # Generate the blank coefficient matrix
        self.coefficientVector = np.zeros([self.numNormalizedInputs,1])

        # Initialize the correlation dictionary
        self.initializeCorrelationDict();

        # All other matrices/vectors are left the same, as they are dependent
        # on the number of observations.

        

    #  Add a new training observation. Requirements: newInputObs must be a
    #  row array of size numInputs. newOutputObs must be a single value.
    def addSingleObservation(self, newInputObs, newOutputObs):
        if (len(newInputObs) == self.numInputs
            and type(newOutputObs) not in (tuple, list)):

            # Prepend 1, to represent the unity coefficients
            newInputObs.insert(0, 1);

            # Generate the total number of input sets based on function order
            # (the normalized input set)
            normalizedInputObs = self.generateNormalizedInputs(newInputObs);

            # You can only add values when there is enough history
            if (len(self.inputHistory) == self.inputHistoryNum and
                len(self.outputHistory) == self.outputHistoryNum):
                
                # Create a list with merged input observations to handle history
                mergedInputObs = [];
                mergedInputObs.extend(normalizedInputObs);
                # Extend with the input history
                for inputHistoryRow in self.inputHistory:
                    mergedInputObs.extend(inputHistoryRow);
                # Extend with the output history
                mergedInputObs.extend(self.outputHistory);

                # Only add non-duplicates
                if (not self.__isADuplicate(normalizedInputObs, newOutputObs)):
                    # TODO: Replace the following code with a general impl.
                    if (self.observationMatrix.shape[0] == 0):
                        self.observationMatrix = np.array([mergedInputObs]);
                        self.outputVector = np.array([newOutputObs]);
                        self.numObservations = 1;
                    else:
                        self.observationMatrix = np.append(
                            self.observationMatrix,
                            np.array([mergedInputObs]),
                            axis=0);
                        self.outputVector = np.append(self.outputVector,
                                                      np.array([newOutputObs]),
                                                      axis=0);
                        self.numObservations += 1;

                # Update the input and output observations matrices (if needed)
                if (self.inputHistoryNum > 0):
                    self.inputHistory.pop();
                    self.inputHistory.append(normalizedInputObs);
                if (self.outputHistoryNum > 0):
                    self.outputHistory.pop();
                    self.outputHistory.append(newOutputObs);

            # If there is not enough history, add the current obs to history.
            else:
                if (len(self.inputHistory) < self.inputHistoryNum):
                    self.inputHistory.append(normalizedInputObs);
                if (len(self.inputHistory) < self.inputHistoryNum):
                    self.outputHistory.append(newOutputObs);

            # Update the correlation coefficient keys
            for i in range(len(normalizedInputObs)):
                self.correlationValues[i].append(normalizedInputObs[i]);
        else:
            print("Wrong dimensions! Expected ",
                  str(self.numInputs),
                  ", got ",
                  str(len(newInputObs)));


    #  Add a set of training observations, with the newInputObsMatrix being a
    #  set of correctly-sized vectors and newOutputVector being a vector of
    #  individual values.
    def addBatchObservations(self, newInputObsMatrix, newOutputVector):
        for newInputVector in newInputObsMatrix:
            outputValue = newOutputVector.pop();
            self.addSingleObservation(newInputVector, outputValue);

    #  Generate the normalized inputs based on function order
    def generateNormalizedInputs(self, inputObs):
        normalizedInputObs = [];
        if (self.functionOrder == 1):
            return inputObs;
        elif (self.functionOrder == 2):
            for i in range(0, len(inputObs)):
                for j in range(0, i+1):
                    normalizedInputObs.append(inputObs[i]*inputObs[j]);
        elif (self.functionOrder == 3):
            for i in range(0, len(inputObs)):
                for j in range(0, i+1):
                    for k in range(0, j+1):
                        normalizedInputObs.append(inputObs[i]*inputObs[j]*inputObs[k]);

        return normalizedInputObs;


    #  Train the coefficients on the existing observation matrix if there are
    #  enough observations.
    def train(self):
        if (self.observationMatrix.shape[0] >= self.numNormalizedInputs):
            #print("Training started");
            self.coefficientVector = \
                np.linalg.lstsq(self.observationMatrix, self.outputVector);
        else:
            print("Observation matrix: ", str(self.observationMatrix.shape[0] ));
            print("Normalized inputs needed: ", str(self.numNormalizedInputs));
            print("Not enough observations to train!");
    
    #  Returns True if the provided input vector and output observation already
    #  exist in the observation matrix, False otherwise
    def __isADuplicate(self, normalizedInputVector, outputObs):
        return False;
        for row in range(0, self.observationMatrix.shape[0]):
            if (np.array_equal(self.observationMatrix[row], normalizedInputVector) \
               and self.outputVector[row] == outputObs):
                return True;
        return False;

    #  Test the trained matrix against the given input observation
    def execute(self, inputObsVector):
        inputObsVector.insert(0, 1);
        inputObsNPVector = np.array(self.generateNormalizedInputs(inputObsVector));
        return np.dot(self.coefficientVector[0],inputObsNPVector);

    # Initialize the correlation coefficient keys
    def initializeCorrelationDict(self):
        for i in range(0, self.numNormalizedInputs):
            self.correlationValues[i] = [];

    #  Basic correlation analysis for each coefficient.
    def getCorrelationValues(self):
        calculatedCorrelations = [];
        for i in range(0, self.numNormalizedInputs):
            calculatedCorrelations.append(self.coefficientVector[0][i]/np.std(self.correlationValues[i]));
        
        return calculatedCorrelations;
