from numpy import recfromcsv
from time import strptime
import matplotlib.pyplot as plt 
from Knn import Knn
import numpy as np
import csv
import time as T
from sklearn.metrics import mean_squared_error


#start = time.time()
#print("hello")
#end = time.time()

#open output file
output=open('test/knnDishwasherTestResult.csv', 'w')
fieldnames = ['real_power', 'predict_power']
writer = csv.DictWriter(output, fieldnames=fieldnames)
writer.writeheader()
#writer.writerow({'real_power': 'Baked', 'predict_power': 'Beans'})

    
csv = recfromcsv('all.csv', delimiter=',')

trainer = Knn(complexity=2, numInputs=7, inputClassifiers=np.empty([7]), outputClassifier=0, appFieldsDict=0);
x_train = [];
y_train = [];
x_predict = [];
y_predict = [];
x_real=[];
y_real=[];

numRow = 96
day_train_start=0
day_train_end=150
#day_predict = 103
day_predict_start=150
day_predict_end = 299
#read in csv and parse data to trainer

for i in range(numRow*day_train_start,numRow*(day_train_end+1)):
#for row in csv:
#date = csv[0][0]
#energy = csv[0][1]
	row = csv[i]
	date=row[0]
	dishwasher=csv[i+1][3]

	date=date.replace("/"," ")
	date=date.replace(":"," ")
	#print date;
#time.struct_time(tm_year=2014, tm_mon=1, tm_mday=1, tm_hour=0, tm_min=15, tm_sec=0, 
#tm_wday=2, tm_yday=1, tm_isdst=-1)
#https://docs.python.org/2/library/time.html
	t=strptime(date, "%m %d %Y %H %M")
	weekday = t[6]; #range [0, 6], Monday is 0
	#print weekday
	hour=t[3]
	minute=t[4]
	sec=t[5]
	time = (t[3]*3600+t[4]*60+t[5])/(24*3600.0)
	x_obs = [time, row[2], row[4], row[5], row[6], row[7], row[8]]
	x_train.append(x_obs)
	y_obs = dishwasher
	y_train.append(y_obs)
 	trainer.addSingleObservation(x_obs, y_obs);

trainStart = T.clock()
trainer.train();
trainEnd = T.clock()
trainLatency = trainEnd-trainStart

testLatency=0
MAE = 0
MBE = 0
for i in range(numRow*day_predict_start,numRow*(day_predict_end+1)):
	row = csv[i]
	date=row[0]
	date_predict = csv[i+1][0]
	dishwasher=round(csv[i+1][3],4)
	date=date.replace("/"," ")
	date=date.replace(":"," ")
	date_predict=date_predict.replace("/"," ")
	date_predict=date_predict.replace(":"," ")
	t=strptime(date, "%m %d %Y %H %M")
	t_predict = strptime(date_predict, "%m %d %Y %H %M")
	weekday = t[6]; #range [0, 6], Monday is 0
	hour=t[3]
	minute=t[4]
	sec=t[5]
	time = (t[3]*3600+t[4]*60+t[5])/(24*3600.0)
	time_predict = (t_predict[3]*3600+t_predict[4]*60+t_predict[5])/(24*3600.0)
	x_predict=[time_predict, row[2], row[4], row[5], row[6], row[7], row[8]];
	testStart = T.clock()
	result = trainer.execute(x_predict);
	result = round(result,4)
	testEnd = T.clock()
	testLatency += testEnd - testStart
	#y_predict = np.concatenate((y_predict,result));
	y_predict.append(result);
	x_real.append([time_predict])
	y_real.append(dishwasher)
	#write result to output file
	writer.writerow({'real_power': dishwasher, 'predict_power': result})
	
	#MAE
	MAE += abs(dishwasher-result);
	MBE += dishwasher-result


output.close();
numTest = numRow*(day_predict_end+1-day_predict_start)
testLatencyAvg = testLatency/numTest

MAE = MAE/numTest
MBE = MBE/numTest
RMSE = mean_squared_error(y_real, y_predict)**0.5

testLog = open('test/knnDishwasherTest-93.all.csv-out.log', 'w');


testLog.write("K-Nearest Neighbor")
testLog.write("\n")
testLog.write("training latency = " + str(trainLatency))
testLog.write("\n")
testLog.write("test latency (total) = "+ str(testLatency))
testLog.write("\n")
testLog.write("test latency (avg) = "+ str(testLatencyAvg))
testLog.write("\n")
testLog.write("MAE = "+ str(MAE))
testLog.write("\n")
testLog.write("RMSE = "+ str(RMSE))
testLog.write("\n")
testLog.write("MBE = "+ str(MBE))
testLog.write("\n")
testLog.close();


"""
plt.subplot(1, 1, 1)
plt.scatter(x_real, y_real, c='r', label='data')
plt.scatter(x_real, y_predict, c='g', label='prediction')
plt.axis('tight')
plt.legend()
plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (2,
	"uniform"))

plt.show()
"""