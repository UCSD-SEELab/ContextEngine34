import numpy as np
import scipy
import math
from Anom import Anom
import Anomaly
import csv




inputs = []
targets = []

with open('Data.csv','rb') as f:
    reader = csv.reader(f)
    csvData = list(reader)

#Set data formats
time = [i[0] for i in csvData] 
data = [i[1] for i in csvData]

convert = 0
while (convert < len(data)):
    data[convert] = float(data[convert])
    convert = convert + 1

np.asarray(data)

mag = np.linalg.norm(data)


newData=np.array([0]*len(data))


anomalyDet = Anom (complexity = 0, numInputs = len(data), outputClassifier = newData, inputClassifiers = data, appFieldsDict = [])


anomalyDet.addSingleObservation(data,newData)

anomalyDet.train()

anomalyDet.execute(data)










