#include "smctc.hh"
#include "pffuncs.hh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string.h>

#include <sys/time.h>
using namespace std;



typedef unsigned long long timestamp_t;

static timestamp_t
get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}


///The observations
cv_obs * y;
long load_data(char const * szName, cv_obs** y);
double integrand_mean_y(const cv_state&, void*);

double prevY = 0;
int sampleIndex =0;
// double integrand_mean_y(const cv_state&, void*);
// double integrand_var_x(const cv_state&, void*);
double integrand_var_y(const cv_state&, void*);

int main(int argc, char** argv)
{
  //Assigning number of particles
  long arg = atol(argv[1]);

  //Setting up the output file
  string fileNameStr = argv[1];
  fileNameStr = fileNameStr +argv[2];

  //Number of iterations.
  long lIterates = atol(argv[3]);

  ofstream outFile;

  outFile.open (fileNameStr.c_str());

  long lNumber = arg;

  try {
    //Load observations
    cout << "Opening "<<argv[2]<<endl;
    load_data(argv[2], &y);
    cout << "Finish data.csv"<<endl;

    //Initialise and run the sampler
    smc::sampler<cv_state> Sampler(lNumber, SMC_HISTORY_NONE);
    smc::moveset<cv_state> Moveset(fInitialise, fMove, NULL);

    timestamp_t t0 = get_timestamp();

    Sampler.SetResampleParams(SMC_RESAMPLE_RESIDUAL, 0.5);

    timestamp_t t1 = get_timestamp();
    // cout << "The Set ResampleParams Ex Time" << (t1 - t0) / 1000000.0L <<endl;

    t0 = get_timestamp();
    Sampler.SetMoveSet(Moveset);
    t1 = get_timestamp();
	//  cout << "The Set Move Set Ex Time" << (t1 - t0) / 1000000.0L <<endl;


    t0 = get_timestamp();
    Sampler.Initialise();
    t1 = get_timestamp();
    //cout << "The Initialise Ex Time" << (t1 - t0) / 1000000.0L <<endl;

    // Process
    double aveIter = 0.0;
    double cur = 0.0;
    double aveInter = 0.0;


    for(int n=1 ; n < lIterates ; ++n) {


      // Iterate moves the particles
      //Check if ltime is the same as y[lindex].xposition.
      //So we can update the index after is used on the particles

      t0 = get_timestamp();
      Sampler.Iterate();
      t1 = get_timestamp();

      cur = ((t1 - t0) / 1000000.0L);
      // cout << "Iter:" << cur << endl;


      aveIter += cur;
      double xm,xv,ym,yv;

      t0 = get_timestamp();
      // xm = Sampler.Integrate(integrand_mean_x,NULL);
      // xv = Sampler.Integrate(integrand_var_x, (void*)&xm);


      ym = Sampler.Integrate(integrand_mean_y,NULL);
      yv = Sampler.Integrate(integrand_var_y, (void*)&ym);
      // printf("Cur var ym %f, Prev ym: %f\n", ym, prevY);

      // Checking if we reached a sample
      //So we can update the index

      if ((n) == y[sampleIndex].x_pos){
        // printf("The sample index in main %d\n", sampleIndex);
        // printf("n: %d, xpos:%f\n",(n), y[sampleIndex].x_pos);

        sampleIndex++;

      }
      prevY = ym;

      t1 = get_timestamp();

      cur = ((t1 - t0) / 1000000.0L);
      aveInter += cur;

      // cout << "Inte:" << (t1 - t0) / 1000000.0L <<"\n " << endl;

      cout << "Sample num:"<< n << " "<< ym << ","<< yv << ","  << endl;

      outFile << n <<","<<ym << "\n";

      //cout << ym<< ",";

    }
    outFile.close();
    cout << "Iter Ave"<< aveIter/lIterates <<endl;
    cout << "Itegrate Ave"<< aveInter/lIterates <<endl;


  }

  catch(smc::exception  e)
    {
      cerr << e;
      exit(e.lCode);
    }
}
long load_data(char const * szName, cv_obs** yp)
{
  FILE * fObs = fopen(szName,"rt");
  if (!fObs)
    throw SMC_EXCEPTION(SMCX_FILE_NOT_FOUND, "Error: pf assumes that the current directory contains an appropriate data file called data.csv\nThe first line should contain a constant indicating the number of data lines it contains.\nThe remaining lines should contain comma-separated pairs of x,y observations.");
  char* szBuffer = new char[1024];
  fgets(szBuffer, 1024, fObs);
  long lIterates = strtol(szBuffer, NULL, 10);

  *yp = new cv_obs[lIterates];

  for(long i = 0; i < lIterates; ++i)
    {
      fgets(szBuffer, 1024, fObs);
      (*yp)[i].x_pos = strtod(strtok(szBuffer, ",\r\n "), NULL);
      (*yp)[i].y_pos = strtod(strtok(NULL, ",\r\n "), NULL);
      // printf("y:%f, x:%f \n", (*yp)[i].y_pos, (*yp)[i].x_pos);

    }
  fclose(fObs);
  printf("%s\n","Done with file" );
  delete [] szBuffer;

  return lIterates;
}

double integrand_mean_y(const cv_state& s, void *)
{
  return s.y_pos;
}

double integrand_var_y(const cv_state& s, void* vmy)
{
  double* dmy = (double*)vmy;
  double d = (s.y_pos - (*dmy));
  return d*d;
}
