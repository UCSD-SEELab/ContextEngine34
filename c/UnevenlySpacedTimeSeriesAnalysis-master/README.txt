Sequential Importance Resampling for Unevenly Spaced Time Series Prediction. 

The repo contains the whole SMCTC: Sequential Monte Carlo Template Class in order
to make the installation and execution faster. 

Compilation and Execution.
1. 
- The GNU Scintific Library is a dependency for SMCTC.
-The first step is to install it with the following command:
sudo apt-get install gsl-bin libgsl0-dev

2.
-To compile first make the libraries by running "make" on the command line. 
This will compile the library and examples. 

3.
-We are ready to compile and run the program by going to the following path:
UnevenlySpacedTimeSeriesAnalysis/pf

-To compile simply run make. 
-this will create a binary called ./pf.

Running the binary:
To run the binary  you have to pass in three command line arguments:
./pf <# of Particles> <CSV file path> <Number of iterations>

The CSV file path will the file used to simulate a data stream. 
The number of iterations define the number of predictions the algorithm will take. 
The algorithm will write the output in CSV format to a file with name <#particles><filename>.

The algorithm checks if there is a field on the CSV file with a time stamp that 
is the same as the current iteration number . 
As an example: 
With a file like , 1 100, 2 200, 5 250, 7 220
The iteration number will start at 1 then check on the csv file if there is a value with time 
1. In this case there is. Then the algorithm will use this value to smooth out the particles. 
If we iterate furthermor and we reach iteration number 3, we can see that there is no value on 
the csv file at time 3. In this case the algorithm will use the previously seen value to predict the 
the value at time 3. 
