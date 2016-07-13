#This is the python version of the S-H-ESD algorithm,
#origininally implemented by twitter in R. 
#Version of S-H-ESD Algorithm from twitter/AnomalyDetection
#Algorithm takes in a set of time series data and runs a modified
#version of the S-H-ESD algorithm, removing anomalies from the dataset

import csv
import numpy as np
import scipy
import math

#Read in data
#with open(filename, 'rb') as f:
#    reader = csv.reader(f)
#    csvData = list(reader)

#Set data formats
#time = [i[0] for i in csvData] 
#data = [i[1] for i in csvData]
#np.asarray(data)





def AnomThresh(data):

    ares = [];
    i = 0;
    while(i < len(data)):
        ares = abs(data[i])
        i = i+1;
    
    datasigma = median(ares);

    i = 0;
    while(i < len(data)):
        ares[i] = ares[i] / datasigma;
        i = i + 1;

    newSigma = median(ares)

    thresh = 3*newSigma
        
    n = len(data)
    
    t = math.sqrt((n-1+1+datasigma**2)*(n-i+1))
    lam = datasigma*(n-1)/t
 
    
    return thresh



def median(ls):
    ls = sorted(ls)
    if(len(ls) < 1):
        return None
    if(len(ls) % 2 == 1):
        return ls[((len(ls)+1)/2)-1]
    else:
        return ls[((len(ls)+1)/2)]

