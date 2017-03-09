#include "smctc.hh"
#include "pffuncs.hh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

double integrand_mean_x(const cv_state&, void*);
double integrand_mean_y(const cv_state&, void*);
double integrand_var_x(const cv_state&, void*);
double integrand_var_y(const cv_state&, void*);

int main(int argc, char** argv)
{
  printf("%s\n",argv[0] );
  long lNumber = 1000;
  long lIterates;

  try {
    //Load observations
    cout << "Opening data.csv"<<endl;
    lIterates = load_data("data.40csv", &y);

    //Initialise and run the sampler
    smc::sampler<cv_state> Sampler(lNumber, SMC_HISTORY_NONE);
    smc::moveset<cv_state> Moveset(fInitialise, fMove, NULL);

    timestamp_t t0 = get_timestamp();

    Sampler.SetResampleParams(SMC_RESAMPLE_RESIDUAL, 0.5);

    timestamp_t t1 = get_timestamp();
    cout << "The Set ResampleParams Ex Time" << (t1 - t0) / 1000000.0L <<endl;

    t0 = get_timestamp();
    Sampler.SetMoveSet(Moveset);
    t1 = get_timestamp();
    cout << "The Set Move Set Ex Time" << (t1 - t0) / 1000000.0L <<endl;


    t0 = get_timestamp();
    Sampler.Initialise();
    t1 = get_timestamp();
    cout << "The Initialise Ex Time" << (t1 - t0) / 1000000.0L <<endl;

    // Process



    for(int n=1 ; n < lIterates ; ++n) {
      t0 = get_timestamp();
      Sampler.Iterate();
      t1 = get_timestamp();
      cout << "The Iterate Ex Time" << (t1 - t0) / 1000000.0L <<"\n " << endl;



      double xm,xv,ym,yv;

      t0 = get_timestamp();
      xm = Sampler.Integrate(integrand_mean_x,NULL);
      xv = Sampler.Integrate(integrand_var_x, (void*)&xm);
      ym = Sampler.Integrate(integrand_mean_y,NULL);
      yv = Sampler.Integrate(integrand_var_y, (void*)&ym);
      t1 = get_timestamp();
      cout << "The Integrate Ex Time" << (t1 - t0) / 1000000.0L <<"\n " << endl;

      cout << "Sample num:"<< n << " "<< xm << "," << ym << "," << xv << "," << yv << endl;
    }
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
      printf("HELLO \n" );
    }
  fclose(fObs);

  delete [] szBuffer;

  return lIterates;
}

double integrand_mean_x(const cv_state& s, void *)
{
  return s.x_pos;
}

double integrand_var_x(const cv_state& s, void* vmx)
{
  double* dmx = (double*)vmx;
  double d = (s.x_pos - (*dmx));
  return d*d;
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
