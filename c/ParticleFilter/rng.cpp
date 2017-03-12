#include <cstring>

//! \file
//! \brief This file contains the untemplated functions used for dealing with random number generation.

#include "rng.hh"
#include "gsl_randist.h" // CSC added
#include "gsl_rng.h" // CSC added
#include "smctc.hh"

/*********************** copied content from gsl files *************/
#define N 624   /* Period parameters */
#define M 397
static inline unsigned long int mt_get (void *vstate);
static double mt_get_double (void *vstate);
static void mt_set (void *state, unsigned long int s);

/* most significant w-r bits */
static const unsigned long UPPER_MASK = 0x80000000UL;   

/* least significant r bits */
static const unsigned long LOWER_MASK = 0x7fffffffUL;   

typedef struct
  {
    unsigned long mt[N];
    int mti;
  }
mt_state_t;

static inline unsigned long
mt_get (void *vstate)
{
  mt_state_t *state = (mt_state_t *) vstate;

  unsigned long k ;
  unsigned long int *const mt = state->mt;

#define MAGIC(y) (((y)&0x1) ? 0x9908b0dfUL : 0)

  if (state->mti >= N)
    {   /* generate N words at one time */
      int kk;

      for (kk = 0; kk < N - M; kk++)
        {
          unsigned long y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
          mt[kk] = mt[kk + M] ^ (y >> 1) ^ MAGIC(y);
        }
      for (; kk < N - 1; kk++)
        {
          unsigned long y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
          mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ MAGIC(y);
        }

      {
        unsigned long y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ MAGIC(y);
      }

      state->mti = 0;
    }

  /* Tempering */
  
  k = mt[state->mti];
  k ^= (k >> 11);
  k ^= (k << 7) & 0x9d2c5680UL;
  k ^= (k << 15) & 0xefc60000UL;
  k ^= (k >> 18);

  state->mti++;

  return k;
}

static double
mt_get_double (void * vstate)
{
  return mt_get (vstate) / 4294967296.0 ;
}

static void
mt_set (void *vstate, unsigned long int s)
{
  mt_state_t *state = (mt_state_t *) vstate;
  int i;

  if (s == 0)
    s = 4357;   /* the default seed is 4357 */

  state->mt[0]= s & 0xffffffffUL;

  for (i = 1; i < N; i++)
    {
      /* See Knuth's "Art of Computer Programming" Vol. 2, 3rd
         Ed. p.106 for multiplier. */

      state->mt[i] =
        (1812433253UL * (state->mt[i-1] ^ (state->mt[i-1] >> 30)) + i);
      
      state->mt[i] &= 0xffffffffUL;
    }

  state->mti = i;
}
static const gsl_rng_type mt_type =
{"mt19937",                     /* name */
 0xffffffffUL,                  /* RAND_MAX  */
 0,                             /* RAND_MIN  */
 sizeof (mt_state_t),
 &mt_set,
 &mt_get,
 &mt_get_double};

const gsl_rng_type *gsl_rng_mt19937 = &mt_type;
const gsl_rng_type *gsl_rng_default = &mt_type;
unsigned long int gsl_rng_default_seed = 0;

gsl_rng *
gsl_rng_alloc (const gsl_rng_type * T)
{

  gsl_rng *r = (gsl_rng *) malloc (sizeof (gsl_rng));

  if (r == 0)
    {
	/*
      GSL_ERROR_VAL ("failed to allocate space for rng struct",
                        GSL_ENOMEM, 0);
	*/
    };

  r->state = malloc (T->size);

  if (r->state == 0)
    {
      free (r);         /* exception in constructor, avoid memory leak */

	/*
      GSL_ERROR_VAL ("failed to allocate space for rng state",
                        GSL_ENOMEM, 0);
	*/
    };

  r->type = T;

  // gsl_rng_set (r, gsl_rng_default_seed);        /* seed the generator */
  gsl_rng_set (r, 0);        /* seed the generator */

  return r;
}

void
gsl_rng_set (const gsl_rng * r, unsigned long int seed)
{
  (r->type->set) (r->state, seed);
}

const gsl_rng_type *
gsl_rng_env_setup (void)
{
  unsigned long int seed = 0;
  const char *p = getenv ("GSL_RNG_TYPE");

  gsl_rng_default = gsl_rng_mt19937;

  p = getenv ("GSL_RNG_SEED");
  gsl_rng_default_seed = seed;

  return gsl_rng_default;
}

 double
gsl_rng_uniform_pos (const gsl_rng * r)
{
  double x ;
  do
    {
      x = (r->type->get_double) (r->state) ;
    }
  while (x == 0) ;

  return x ;
}


/* Polar (Box-Mueller) method; See Knuth v2, 3rd ed, p122 */

double
gsl_ran_gaussian (const gsl_rng * r, const double sigma)
{
  double x, y, r2; 

  do  
    {   
      /* choose x,y in uniform square (-1,-1) to (+1,+1) */
      x = -1 + 2 * gsl_rng_uniform_pos (r);
      y = -1 + 2 * gsl_rng_uniform_pos (r);

      /* see if it is in the unit circle */
      r2 = x * x + y * y;
    }   
  while (r2 > 1.0 || r2 == 0); 

  /* Box-Muller transform */
  return sigma * y * sqrt (-2.0 * log (r2) / r2);
}

/*********************** END copied content from gsl files *************/

namespace smc {
    ///The GSL provides a mechanism for obtaining a list of available random number generators.
    ///
    ///This class provides a wrapper for this mechanism and makes it simple to implement software which allows
    ///the nature of the random number generator to be specified at runtime (and to allow it to be a user-specifiable
    ///parameter).
    ///
    ///For example, gslrnginfo::GetNumber can be used to determine how many RNGs are available and gslrnginfo::GetNameByIndex
    ///can then be used to populate a list box with their names. Once the user has selected on the gslrnginfo::GetPointerByName
    ///function can be used to obtain a pointer to the appropriate type and this can be used to produce a random number generator
    ///of the desired type.
    ///
    ///There should be exactly one instance of this class in any program, and that instance is created by
    ///the library. A singleton DP implementation ensures that any additional attempt to instatiate the class simply returns 
    ///a reference to the existing instance.

	/*
    gslrnginfo::gslrnginfo()
    {
	typePtArray = gsl_rng_types_setup();

	const gsl_rng_type** ptIndex = typePtArray;
	nNumber = 0;
	while(ptIndex[0]) {
	    ptIndex++;
	    nNumber++;
	}
	return;
    }

    ///Returns a pointer to a single instance of this class.
    gslrnginfo* gslrnginfo::GetInstance()
    {
	static gslrnginfo ginfo;

	return &ginfo;
    }

    ///This function returns the number of available generators
    int gslrnginfo::GetNumber(void)
    {
	return nNumber;
    }

    ///This function returns the name of the specified generator
    const char* gslrnginfo::GetNameByIndex(int nIndex)
    {
	if(0 <= nIndex && nIndex < nNumber)
	    return typePtArray[nIndex]->name;
	return NULL;
    }

    ///This function returns a pointer to the specified generator type
    const gsl_rng_type* gslrnginfo::GetPointerByIndex(int nIndex)
    {
	if(0 <= nIndex && nIndex < nNumber)
	    return typePtArray[nIndex];
	return NULL;
    }

    ///This function returns a pointer to the specified generator type
    const gsl_rng_type* gslrnginfo::GetPointerByName(const char* szName)
    {
	for(int n = 0; n < nNumber; n++)
	    if(!strcmp(typePtArray[n]->name, szName))
		return typePtArray[n];

	return NULL;
    }
	*/


    ///When called without any arguments, the constructor for the smc::rng class simply allocates a buffer for a
    ///random number generator of type gsl_rng_default (something which can be set at run-time via an environment
    ///variable) using its default seed (which again can be over-ridden using an environment variable).
    rng::rng(void)
    {
	gsl_rng_env_setup();
	type = gsl_rng_default;
	pWorkspace = gsl_rng_alloc(gsl_rng_default);
    }

    ///When called with a single argument, the constructor for the smc::rng class allocates a buffer for a
    ///random number generator of the specified type and initialises it with the default seed (which can be set using
    ///and environment variable if one wishes to vary it at run-time).
    ///
    ///\param Type The type of a GSL random number generator
    rng::rng(const gsl_rng_type* Type)
    {
	gsl_rng_env_setup();
	type = Type;
	pWorkspace = gsl_rng_alloc(Type);
    }

    ///When called with a pair of arguments, the constructor for the smc::rng class allocates a buffer for the specified
    ///random number generator type and initialises it with the specified seed (note that zero has special significance and
    ///is used to specify the seed with which the generator was originally used).
    ///
    ///\param Type The type of a GSL random number generator
    ///\param lSeed The value with which the generator is to be seeded
    rng::rng(const gsl_rng_type* Type, unsigned long int lSeed)
    {
	gsl_rng_env_setup();
	type = Type;
	pWorkspace = gsl_rng_alloc(Type);
	gsl_rng_set(pWorkspace, lSeed);
    }

    ///The destructor presently does no more than call the gsl_rng_free function to deallocate the memory which was
    ///previously allocate to the random number generator.
    rng::~rng()
    {
	// gsl_rng_free(pWorkspace);    
	free(pWorkspace->state);
	free(pWorkspace);
    }

    ///This function returns a pointer to the underlying GSL random number generator which may be used to provide random
    ///number facilities which are not explicitly provided by the intermediate layer of smc::rng.
    gsl_rng* rng::GetRaw(void)
    {
	return pWorkspace;
    }

    ///This function simply passes the relevant arguments on to gsl_ran_multinomial.
    ///     \param n Number of entities to assign.
    ///     \param k Number of categories.
    ///     \param w Weights of category elements
    ///     \param X Array in which to return the sample values.
    void rng::Multinomial(unsigned n, unsigned k, const double* w, unsigned* X)
    {
	gsl_ran_multinomial(pWorkspace, k, n, w, X);
    }


    ///This function simply calls gsl_rng_uniform_int and shifts the
    /// result as appropriate such that the result is an integer generated uniformly from those between
    /// the two arguments (inclusive of those points).
    ///
    ///     \param lMin The smallest value which can be returned
    ///      \param lMax the largest value which can be returned
	/*
    long rng::UniformDiscrete(long lMin, long lMax)
    {
	return gsl_rng_uniform_int(pWorkspace, lMax - lMin + 1) + lMin;
    }
	*/

    ///This function simply calls gsl_ran_beta with the specified parameters.
    ///     \param da The parameter associated with "x".
    ///     \paran db The parameter associated with "1-x".
	/*
    double rng::Beta(double da, double db)
    {
	return gsl_ran_beta(pWorkspace, da, db);
    }
	*/

    ///This function simply calls gsl_ran_cauchy with the specified parameter.
    ///     \param dScale The scale parameter of the distribution.
	/*
    double rng::Cauchy(double dScale)
    {
	return gsl_ran_cauchy(pWorkspace, dScale);
    }
	*/

    ///This function simply calls gsl_ran_exponential with the specified parameters.
    ///     \param dMean The scale (not rate) (and mean) of the distribution.
	/*
    double rng::Exponential(double dMean)
    {
	return gsl_ran_exponential(pWorkspace, dMean);
    }
	*/

    ///This function simply calls gsl_ran_gamma with the specified parameters.
    ///     \param dAlpha The shape of the distribution (integers lead to Erlang distributions)
    ///     \param dBeta The scale (not rate) of the distribution.
	/*
    double rng::Gamma(double dAlpha, double dBeta)
    {
	return gsl_ran_gamma(pWorkspace, dAlpha, dBeta);
    }
	*/

    ///This function simply calls gsl_ran_laplace with the specified parameters.
    ///     \param dScale The scale (not rate) of the distribution.
    ///
	/*
    double rng::Laplacian(double dScale)
    {
	return gsl_ran_laplace(pWorkspace, dScale);
    }
	*/

    ///This function simply calls gsl_ran_lognormal with the specified parameters.
    ///     \param dMu The location parameter of the distribution.
    ///     \param dSigma The scale parameter of the distribution.
	/*
    double rng::Lognormal(double dMu, double dSigma)
    {
	return gsl_ran_lognormal(pWorkspace, dMu, dSigma);
    }
	*/

    ///This function simply calls gsl_ran_gaussian with the specified standard deviation and shifts the result.
    ///     \param dMean The mean of the distribution.
    ///     \param dStd  The standard deviation of the distribution
    double rng::Normal(double dMean, double dStd)
    {
	return dMean + gsl_ran_gaussian(pWorkspace, dStd);
    }
  
    ///This function simply calls gsl_ran_ugaussian returns the result. 
	/*
    double rng::NormalS(void)
    {
	return gsl_ran_ugaussian(pWorkspace);
    }
	*/

    ///This function simply calls gsl_ran_gaussian_tail with the specified parameters and performs appropriate shifting.
    ///     \param dMean The mean of the distribution.
    ///     \param dStd  The standard deviation of the distribution
    ///     \param dThreshold The lower truncation threshold.
	/*
    double rng::NormalTruncated(double dMean, double dStd, double dThreshold)
    {
	return dMean + gsl_ran_gaussian_tail(pWorkspace, dThreshold - dMean, dStd);
    }
	*/

    ///This function simply calls gsl_ran_tdist with the specified number of degrees of freedom.
    ///     \param dDF The number of degrees of freedom.
	/*
    double rng::StudentT(double dDF)
    {
	return gsl_ran_tdist(pWorkspace, dDF);
    }
	*/

    ///This function simply calls gsl_rng_uniform and scales and shifts appropriately. 
    ///     \param dMin The lowest value with positive density.
    ///     \param dMax The largest value with positive density.
	/*
    double rng::Uniform(double dMin, double dMax)
    {
	double rValue;
 
	rValue = gsl_rng_uniform(pWorkspace);
	rValue *= (dMax - dMin);
	rValue += dMin;

	return rValue;
    }
	*/

    ///This function simply calls gsl_rng_uniform. 
	/*
    double rng::UniformS(void) {
	return gsl_rng_uniform(pWorkspace); 
    }
	*/
}
