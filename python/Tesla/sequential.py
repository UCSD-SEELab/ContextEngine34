import numpy as np
import os.path
import pickle
import math
from Tesla import Tesla



# Tesla training objects
kitchOut2Trainer = Tesla(complexity=2,numInputs=1);
kitchOut3Trainer = Tesla(complexity=2,numInputs=1);
kitchOut4Trainer = Tesla(complexity=2,numInputs=1);
microwaveTrainer = Tesla(complexity=2,numInputs=1);
oven01Trainer = Tesla(complexity=2,numInputs=1);
oven02Trainer = Tesla(complexity=2,numInputs=1);
refrigeratorTrainer = Tesla(complexity=2,numInputs=1);
stoveTrainer = Tesla(complexity=2,numInputs=1);

# Pickle files for trained data
kitchOut2PickleFilename = 'traces/kitch_out2_trained_order' + str(kitchOut2Trainer.functionOrder) + '.dat';
kitchOut3PickleFilename = 'traces/kitch_out3_trained_order' + str(kitchOut3Trainer.functionOrder) + '.dat';
kitchOut4PickleFilename = 'traces/kitch_out4_trained_order' + str(kitchOut4Trainer.functionOrder) + '.dat';
microwavePickleFilename = 'traces/microwave_trained_order' + str(microwaveTrainer.functionOrder) + '.dat';
oven01PickleFilename = 'traces/oven01_trained_order' + str(oven01Trainer.functionOrder) + '.dat';
oven02PickleFilename = 'traces/oven02_trained_order' + str(oven02Trainer.functionOrder) + '.dat';
refrigeratorPickleFilename = 'traces/refrigerator_trained_order' + str(refrigeratorTrainer.functionOrder) + '.dat';
stovePickleFilename = 'traces/stove_trained_order' + str(stoveTrainer.functionOrder) + '.dat';

# Open the files needed: input trace and output observations
kitchOut2Trace = open('traces/kitch_out2.dat', mode='r');
kitchOut3Trace = open('traces/kitch_out3.dat', mode='r');
kitchOut4Trace = open('traces/kitch_out4.dat', mode='r');
microwaveTrace = open('traces/microwave.dat', mode='r');
oven01Trace = open('traces/oven01.dat', mode='r');
oven02Trace = open('traces/oven02.dat', mode='r');
refrigeratorTrace = open('traces/refrigerator.dat', mode='r');
stoveTrace = open('traces/stove.dat', mode='r');

kitchOut2OutputTrace = open('traces/kitch_out2_obs.dat', mode='r');
kitchOut3OutputTrace = open('traces/kitch_out3_obs.dat', mode='r');
kitchOut4OutputTrace = open('traces/kitch_out4_obs.dat', mode='r');
microwaveOutputTrace = open('traces/microwave_obs.dat', mode='r');
oven01OutputTrace = open('traces/oven01_obs.dat', mode='r');
oven02OutputTrace = open('traces/oven02_obs.dat', mode='r');
refrigeratorOutputTrace = open('traces/refrigerator_obs.dat', mode='r');
stoveOutputTrace = open('traces/stove_obs.dat', mode='r');


# If we don't have trained Tesla objects already, we need to retrain
if (not os.path.isfile(kitchOut2PickleFilename \
                       and kitchOut3PickleFilename \
                       and kitchOut4PickleFilename \
                       and microwavePickleFilename \
                       and oven01PickleFilename \
                       and oven02PickleFilename \
                       and refrigeratorPickleFilename \
                       and stovePickleFilename)):
    # Add observations to the appropriate tesla training objects
    for i in range(0, 10000):
        # Kitchen Outlets 2
        (timestamp, kitchOut2Entry) = kitchOut2Trace.readline().split();
        kitchOut2Entry = float(kitchOut2Entry);
        (timestamp, kitchOut2Output) = kitchOut2OutputTrace.readline().split();
        kitchOut2Output = int(kitchOut2Output);
        kitchOut2Trainer.addSingleObservation([kitchOut2Entry], kitchOut2Output);

        # Kitchen Outlets 3
        (timestamp, kitchOut3Entry) = kitchOut3Trace.readline().split();
        kitchOut3Entry = float(kitchOut3Entry);
        (timestamp, kitchOut3Output) = kitchOut3OutputTrace.readline().split();
        kitchOut3Output = int(kitchOut3Output);
        kitchOut3Trainer.addSingleObservation([kitchOut3Entry], kitchOut3Output);

        # Kitchen Outlets 4
        (timestamp, kitchOut4Entry) = kitchOut4Trace.readline().split();
        kitchOut4Entry = float(kitchOut4Entry);
        (timestamp, kitchOut4Output) = kitchOut4OutputTrace.readline().split();
        kitchOut4Output = int(kitchOut4Output);
        kitchOut4Trainer.addSingleObservation([kitchOut4Entry], kitchOut4Output);

        # Microwave
        (timestamp, microwaveEntry) = microwaveTrace.readline().split();
        microwaveEntry = float(microwaveEntry);
        (timestamp, microwaveOutput) = microwaveOutputTrace.readline().split();
        microwaveOutput = int(microwaveOutput);
        microwaveTrainer.addSingleObservation([microwaveEntry], microwaveOutput);

        # Oven 01
        (timestamp, oven01Entry) = oven01Trace.readline().split();
        oven01Entry = float(oven01Entry);
        (timestamp, oven01Output) = oven01OutputTrace.readline().split();
        oven01Output = int(oven01Output);
        oven01Trainer.addSingleObservation([oven01Entry], oven01Output);

        # Oven 02
        (timestamp, oven02Entry) = oven02Trace.readline().split();
        oven02Entry = float(oven02Entry);
        (timestamp, oven02Output) = oven02OutputTrace.readline().split();
        oven02Output = int(oven02Output);
        oven02Trainer.addSingleObservation([oven02Entry], oven02Output);

        # Refrigerator
        (timestamp, refrigeratorEntry) = refrigeratorTrace.readline().split();
        refrigeratorEntry = float(refrigeratorEntry);
        (timestamp, refrigeratorOutput) = refrigeratorOutputTrace.readline().split();
        refrigeratorOutput = int(refrigeratorOutput);
        refrigeratorTrainer.addSingleObservation([refrigeratorEntry], refrigeratorOutput);

        # Stove
        (timestamp, stoveEntry) = stoveTrace.readline().split();
        stoveEntry = float(stoveEntry);
        (timestamp, stoveOutput) = stoveOutputTrace.readline().split();
        stoveOutput = int(stoveOutput);
        stoveTrainer.addSingleObservation([stoveEntry], stoveOutput);

    # Train the individual appliances.
    kitchOut2Trainer.train();
    kitchOut3Trainer.train();
    kitchOut4Trainer.train();
    microwaveTrainer.train();
    oven01Trainer.train();
    oven02Trainer.train();
    refrigeratorTrainer.train();
    stoveTrainer.train();

    # Pickle the Tesla objects, to prevent them from needing to be trained again.
    kitchOut2PickleFileRef = open(kitchOut2PickleFilename, 'wb');
    pickle.dump(kitchOut2Trainer, kitchOut2PickleFileRef);

    kitchOut3PickleFileRef = open(kitchOut3PickleFilename, 'wb');
    pickle.dump(kitchOut3Trainer, kitchOut3PickleFileRef);

    kitchOut4PickleFileRef = open(kitchOut4PickleFilename, 'wb');
    pickle.dump(kitchOut4Trainer, kitchOut4PickleFileRef);

    microwavePickleFileRef = open(microwavePickleFilename, 'wb');
    pickle.dump(microwaveTrainer, microwavePickleFileRef);

    oven01PickleFileRef = open(oven01PickleFilename, 'wb');
    pickle.dump(oven01Trainer, oven01PickleFileRef);

    oven02PickleFileRef = open(oven02PickleFilename, 'wb');
    pickle.dump(oven02Trainer, oven02PickleFileRef);

    refrigeratorPickleFileRef = open(refrigeratorPickleFilename, 'wb');
    pickle.dump(refrigeratorTrainer, refrigeratorPickleFileRef);

    stovePickleFileRef = open(stovePickleFilename, 'wb');
    pickle.dump(stoveTrainer, stovePickleFileRef);

# Else, just load the trained data from the pickled files
else:
    kitchOut2PickleFileRef = open(kitchOut2PickleFilename, 'rb');
    kitchOut2Trainer = pickle.load(kitchOut2PickleFileRef);

    kitchOut3PickleFileRef = open(kitchOut3PickleFilename, 'rb');
    kitchOut3Trainer = pickle.load(kitchOut3PickleFileRef);

    kitchOut4PickleFileRef = open(kitchOut4PickleFilename, 'rb');
    kitchOut4Trainer = pickle.load(kitchOut4PickleFileRef);

    microwavePickleFileRef = open(microwavePickleFilename, 'rb');
    microwaveTrainer = pickle.load(microwavePickleFileRef);

    oven01PickleFileRef = open(oven01PickleFilename, 'rb');
    oven01Trainer = pickle.load(oven01PickleFileRef);

    oven02PickleFileRef = open(oven02PickleFilename, 'rb');
    oven02Trainer = pickle.load(oven02PickleFileRef);

    refrigeratorPickleFileRef = open(refrigeratorPickleFilename, 'rb');
    refrigeratorTrainer = pickle.load(refrigeratorPickleFileRef);

    stovePickleFileRef = open(stovePickleFilename, 'rb');
    stoveTrainer = pickle.load(stovePickleFileRef);

    # Iterate through the traces to reach the test set.
    for i in range(0, 40000):
        kitchOut2Trace.readline();
        kitchOut2OutputTrace.readline();
        
        kitchOut3Trace.readline();
        kitchOut3OutputTrace.readline();
        
        kitchOut4Trace.readline();
        kitchOut4OutputTrace.readline();
        
        microwaveTrace.readline();
        microwaveOutputTrace.readline();
        
        oven01Trace.readline();
        oven01OutputTrace.readline();
        
        oven02Trace.readline();
        oven02OutputTrace.readline();
        
        refrigeratorTrace.readline();
        refrigeratorOutputTrace.readline();
        
        stoveTrace.readline();
        stoveOutputTrace.readline();
        

##else:
##    pickleFileRef = open(fridgePickleFile, 'rb');
##    fridgeTrainer = pickle.load(pickleFileRef);


##print(refrigeratorTrainer1.numObservations);
##print(refrigeratorTrainer1.coefficientVector[0]);

##print(refrigeratorTrainer2.numObservations);
##print(refrigeratorTrainer2.coefficientVector[0]);

##print(refrigeratorTrainer3.numObservations);
##print(refrigeratorTrainer3.coefficientVector[0]);

##print(refrigeratorTrainer4.numObservations);
##print(refrigeratorTrainer4.coefficientVector[0]);

# Keep track of accuracy
kitchOut2Error = 0.0;
kitchOut2NumObs = 0.0;
kitchOut2Min = 0.0;
kitchOut2Max = 0.0;

kitchOut3Error = 0.0;
kitchOut3NumObs = 0.0;
kitchOut3Min = 0.0;
kitchOut3Max = 0.0;

kitchOut4Error = 0.0;
kitchOut4NumObs = 0.0;
kitchOut4Min = 0.0;
kitchOut4Max = 0.0;

microwaveError = 0.0;
microwaveNumObs = 0.0;
microwaveMin = 0.0;
microwaveMax = 0.0;

oven01Error = 0.0;
oven01NumObs = 0.0;
oven01Min = 0.0;
oven01Max = 0.0;

oven02Error = 0.0;
oven02NumObs = 0.0;
oven02Min = 0.0;
oven02Max = 0.0;

refrigeratorError = 0.0;
refrigeratorNumObs = 0.0;
refrigeratorMin = 0.0;
refrigeratorMax = 0.0;

stoveError = 0.0;
stoveNumObs = 0.0;
stoveMin = 0.0;
stoveMax = 0.0;

for i in range(0,10000):
    kitchOut2InputObs = float(kitchOut2Trace.readline().split()[1]);
    kitchOut2OutputObs = float(kitchOut2OutputTrace.readline().split()[1]);
    kitchOut2ActualObs = round(kitchOut2Trainer.test([kitchOut2InputObs]));
    kitchOut2ObsError = (kitchOut2ActualObs - kitchOut2OutputObs);
##    if (kitchOut2ObsError > 0.5):
##        print((kitchOut2ActualObs, kitchOut2OutputObs));
    kitchOut2Error += abs(kitchOut2ObsError);
    kitchOut2NumObs += 1;
    kitchOut2Min = kitchOut2ObsError \
                   if (kitchOut2ObsError < kitchOut2Min) \
                   else kitchOut2Min;
    kitchOut2Max = kitchOut2ObsError \
                   if (kitchOut2ObsError > kitchOut2Max) \
                   else kitchOut2Max;
##    print(kitchOut2ActualObs, kitchOut2OutputObs);
    
    kitchOut3InputObs = float(kitchOut3Trace.readline().split()[1]);
    kitchOut3OutputObs = float(kitchOut3OutputTrace.readline().split()[1]);
    kitchOut3ActualObs = round(kitchOut3Trainer.test([kitchOut3InputObs]));
    kitchOut3ObsError = (kitchOut3ActualObs - kitchOut3OutputObs);
    kitchOut3Error += abs(kitchOut3ObsError);
    kitchOut3NumObs += 1;
    kitchOut3Min = kitchOut3ObsError \
                   if (kitchOut3ObsError < kitchOut3Min) \
                   else kitchOut3Min;
    kitchOut3Max = kitchOut3ObsError \
                   if (kitchOut3ObsError > kitchOut3Max) \
                   else kitchOut3Max;

##    print((kitchOut3Trainer.test([kitchOut3InputObs])), kitchOut3OutputObs);

    kitchOut4InputObs = float(kitchOut4Trace.readline().split()[1]);
    kitchOut4OutputObs = float(kitchOut4OutputTrace.readline().split()[1]);
    kitchOut4ActualObs = round(kitchOut4Trainer.test([kitchOut4InputObs]));
    kitchOut4ObsError = (kitchOut4ActualObs - kitchOut4OutputObs);
    kitchOut4Error += abs(kitchOut4ObsError);
    kitchOut4NumObs += 1;
    kitchOut4Min = kitchOut4ObsError \
                   if (kitchOut4ObsError < kitchOut4Min) \
                   else kitchOut4Min;
    kitchOut4Max = kitchOut4ObsError \
                   if (kitchOut4ObsError > kitchOut4Max) \
                   else kitchOut4Max;
##    print((kitchOut4Trainer.test([kitchOut4InputObs])), kitchOut4OutputObs);

    microwaveInputObs = float(microwaveTrace.readline().split()[1]);
    microwaveOutputObs = float(microwaveOutputTrace.readline().split()[1]);
    microwaveActualObs = round(microwaveTrainer.test([microwaveInputObs]));
    microwaveObsError = (microwaveActualObs - microwaveOutputObs);
    microwaveError += abs(microwaveObsError);
    microwaveNumObs += 1;
    microwaveMin = microwaveObsError \
                   if (microwaveObsError < microwaveMin) \
                   else microwaveMin;
    microwaveMax = microwaveObsError \
                   if (microwaveObsError > microwaveMax) \
                   else microwaveMax;
##    print((microwaveTrainer.test([microwaveInputObs])), microwaveOutputObs);

    oven01InputObs = float(oven01Trace.readline().split()[1]);
    oven01OutputObs = float(oven01OutputTrace.readline().split()[1]);
    oven01ActualObs = round(oven01Trainer.test([oven01InputObs]));
    oven01ObsError = (oven01ActualObs - oven01OutputObs);
    oven01Error += abs(oven01ObsError);
    oven01NumObs += 1;
    oven01Min = oven01ObsError \
                   if (oven01ObsError < oven01Min) \
                   else oven01Min;
    oven01Max = oven01ObsError \
                   if (oven01ObsError > oven01Max) \
                   else oven01Max;
##    print((oven01InputObs, oven01Trainer.test([oven01InputObs])));

    oven02InputObs = float(oven02Trace.readline().split()[1]);
    oven02OutputObs = float(oven02OutputTrace.readline().split()[1]);
    oven02ActualObs = round(oven02Trainer.test([oven02InputObs]));
    oven02ObsError = (oven02ActualObs - oven02OutputObs);
    oven02Error += abs(oven02ObsError);
    oven02NumObs += 1;
    oven02Min = oven02ObsError \
                   if (oven02ObsError < oven02Min) \
                   else oven02Min;
    oven02Max = oven02ObsError \
                   if (oven02ObsError > oven02Max) \
                   else oven02Max;
##    print((oven02InputObs, oven02Trainer.test([oven02InputObs])));

    refrigeratorInputObs = float(refrigeratorTrace.readline().split()[1]);
    refrigeratorOutputObs = float(refrigeratorOutputTrace.readline().split()[1]);
    refrigeratorActualObs = round(refrigeratorTrainer.test([refrigeratorInputObs]));
    refrigeratorObsError = (refrigeratorActualObs - refrigeratorOutputObs);
    refrigeratorError += abs(refrigeratorObsError);
    refrigeratorNumObs += 1;
    refrigeratorMin = refrigeratorObsError \
                   if (refrigeratorObsError < refrigeratorMin) \
                   else refrigeratorMin;
    refrigeratorMax = refrigeratorObsError \
                   if (refrigeratorObsError > refrigeratorMax) \
                   else refrigeratorMax;
##    print((refrigeratorInputObs, refrigeratorTrainer.test([refrigeratorInputObs])));

    stoveInputObs = float(stoveTrace.readline().split()[1]);
    stoveOutputObs = float(stoveOutputTrace.readline().split()[1]);
    stoveActualObs = round(stoveTrainer.test([stoveInputObs]));
    stoveObsError = (stoveActualObs - stoveOutputObs);
    stoveError += abs(stoveObsError);
    stoveNumObs += 1;
    stoveMin = stoveObsError \
                   if (stoveObsError < stoveMin) \
                   else stoveMin;
    stoveMax = stoveObsError \
                   if (stoveObsError > stoveMax) \
                   else stoveMax;
##    print((stoveInputObs, stoveTrainer.test([stoveInputObs])));

kitchOut2Trace.close();
kitchOut3Trace.close();
kitchOut4Trace.close();
microwaveTrace.close();
oven01Trace.close();
oven02Trace.close();
refrigeratorTrace.close();
stoveTrace.close();

kitchOut2OutputTrace.close();
kitchOut3OutputTrace.close();
kitchOut4OutputTrace.close();
microwaveOutputTrace.close();
oven01OutputTrace.close();
oven02OutputTrace.close();
refrigeratorOutputTrace.close();
stoveOutputTrace.close();

kitchOut2PickleFileRef.close();
kitchOut3PickleFileRef.close();
kitchOut4PickleFileRef.close();
microwavePickleFileRef.close();
oven01PickleFileRef.close();
oven02PickleFileRef.close();
refrigeratorPickleFileRef.close();
stovePickleFileRef.close();


print("KitchOut2 MAE error: " + str(kitchOut2Error));#/kitchOut2NumObs));
print("Range low: " + str(kitchOut2Min));
print("Range high: " + str(kitchOut2Max));
print("\n");

print("kitchOut3 MAE error: " + str(kitchOut3Error));#/kitchOut3NumObs));
print("Range low: " + str(kitchOut3Min));
print("Range high: " + str(kitchOut3Max));
print("\n");

print("kitchOut4 MAE error: " + str(kitchOut4Error));#/kitchOut4NumObs));
print("Range low: " + str(kitchOut4Min));
print("Range high: " + str(kitchOut4Max));
print("\n");

print("microwave MAE error: " + str(microwaveError));#/microwaveNumObs));
print("Range low: " + str(microwaveMin));
print("Range high: " + str(microwaveMax));
print("\n");

print("oven01 MAE error: " + str(oven01Error));#/oven01NumObs));
print("Range low: " + str(oven01Min));
print("Range high: " + str(oven01Max));
print("\n");

print("oven02 MAE error: " + str(oven02Error));#/oven02NumObs));
print("Range low: " + str(oven02Min));
print("Range high: " + str(oven02Max));
print("\n");

print("refrigerator MAE error: " + str(refrigeratorError));#/refrigeratorNumObs));
print("Range low: " + str(refrigeratorMin));
print("Range high: " + str(refrigeratorMax));
print("\n");

print("stove MAE error: " + str(stoveError));#/stoveNumObs));
print("Range low: " + str(stoveMin));
print("Range high: " + str(stoveMax));
print("\n");


##if (not os.path.isfile(fridgeTraceFile)):
##    for i in range(0, 86401):
##        (timestamp, entry) = refridgeTrace.readline().split();
##        timestamp = int(timestamp);
##        entry = float(entry);
##        outputEntry = 0;
##
##        if (entry > 500):
##    ##        print("High output found!");
##            outputEntry = 1;
##            
##        fridgeTrainer.addSingleObservation([entry], outputEntry);
##
##    fridgeTrainer.train();
##
##    pickleFileRef = open(fridgePickleFile, 'wb');
##    pickle.dump(fridgeTrainer, pickleFileRef);
##
##else:
##    pickleFileRef = open(fridgePickleFile, 'rb');
##    fridgeTrainer = pickle.load(pickleFileRef);
##
##
##print(fridgeTrainer.numObservations);
##print(fridgeTrainer.coefficientVector[0]);
##
##for i in range(0,10001):
##    testObs = float(f.readline().split()[1]);
##
##    if (testObs > 300):
##        print((testObs, fridgeTrainer.test([testObs])));
##
##refrigeratorTrace.close();
##pickleFileRef.close();
##
