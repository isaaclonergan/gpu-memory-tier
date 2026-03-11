/* 
    GPU Benchmark
    Heavily inspired, and most code copied, from STREAM by John D. McCalpin
*/

# include <stdio.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
#include <cuda_runtime.h>
# include <stdlib.h>
# include <string.h>
#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif
#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif
#ifndef OFFSET
#   define OFFSET	0
#endif


# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while(0)

static double	avgtime[4] = {0}, maxtime[4] = {0},
		mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ",
    "Add:       ", "Triad:     "};

static double	bytes[4] = {
        2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
        2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
        3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
        3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE
    };

extern double mysecond();
extern void checkSTREAMresults();

extern void kernel_copy(STREAM_TYPE * __restrict__ c,
                         const STREAM_TYPE * __restrict__ a, size_t n);

extern void kernel_scale(STREAM_TYPE * __restrict__ b,
                          const STREAM_TYPE * __restrict__ c,
                          STREAM_TYPE scalar, size_t n);

extern void kernel_add(STREAM_TYPE * __restrict__ c,
                        const STREAM_TYPE * __restrict__ a,
                        const STREAM_TYPE * __restrict__ b, size_t n);

extern void kernel_triad(STREAM_TYPE * __restrict__ a,
                          const STREAM_TYPE * __restrict__ b,
                          const STREAM_TYPE * __restrict__ c,
                          STREAM_TYPE scalar, size_t n);

int 
main()
{
    int			quantum, checktick();
    int			BytesPerWord;
    int			k;
    ssize_t		j;
    STREAM_TYPE		scalar;
    double		t, times[4][NTIMES];

    /* --- SETUP --- determine precision and check timing --- */

    printf(HLINE);
    printf("STREAM version $Revision: 5.10 $\n");
    printf(HLINE);
    BytesPerWord = sizeof(STREAM_TYPE);
    printf("This system uses %d bytes per array element.\n",
	BytesPerWord);

    printf(HLINE);
#ifdef N
    printf("*****  WARNING: ******\n");
    printf("      It appears that you set the preprocessor variable N when compiling this code.\n");
    printf("      This version of the code uses the preprocessor variable STREAM_ARRAY_SIZE to control the array size\n");
    printf("      Reverting to default value of STREAM_ARRAY_SIZE=%llu\n",(unsigned long long) STREAM_ARRAY_SIZE);
    printf("*****  WARNING: ******\n");
#endif

    printf("Array size = %llu (elements), Offset = %d (elements)\n" , (unsigned long long) STREAM_ARRAY_SIZE, OFFSET);
    printf("Memory per array = %.1f MiB (= %.1f GiB).\n", 
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0),
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0/1024.0));
    printf("Total memory required = %.1f MiB (= %.1f GiB).\n",
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.),
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024./1024.));
    printf("Each kernel will be executed %d times.\n", NTIMES);
    printf(" The *best* time for each kernel (excluding the first iteration)\n"); 
    printf(" will be used to compute the reported bandwidth.\n");

#ifdef _OPENMP
    printf(HLINE);
#pragma omp parallel 
    {
#pragma omp master
	{
	    k = omp_get_num_threads();
	    printf ("Number of Threads requested = %i\n",k);
        }
    }
#endif
    
    size_t array_n     = (size_t)STREAM_ARRAY_SIZE + OFFSET;
    size_t array_bytes = array_n * sizeof(STREAM_TYPE);

    int deviceId = 0;
    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    CUDA_CHECK(cudaSetDevice(deviceId));

    STREAM_TYPE *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, array_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, array_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, array_bytes));

    STREAM_TYPE *ha, *hb, *hc;
    CUDA_CHECK(cudaMallocHost((void**)&ha, array_bytes));
    CUDA_CHECK(cudaMallocHost((void**)&hb, array_bytes));
    CUDA_CHECK(cudaMallocHost((void**)&hc, array_bytes));

#ifdef _OPENMP
	k = 0;
#pragma omp parallel
#pragma omp atomic 
		k++;
    printf ("Number of Threads counted = %i\n",k);
#endif

/* Get initial value for system clock. */
#pragma omp parallel for
    for (j=0; j<STREAM_ARRAY_SIZE; j++) {
	    ha[j] = 1.0;
	    hb[j] = 2.0;
	    hc[j] = 0.0;
	}

    printf(HLINE);

    if  ( (quantum = checktick()) >= 1) 
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
    else {
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");
	quantum = 1;
    }

#pragma omp parallel for
    for (j=0; j<array_n; j++) {
        ha[j] = 1.0;
        hb[j] = 2.0;
        hc[j] = 0.0;
    }
    CUDA_CHECK(cudaMemcpy(d_a, ha, array_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, hb, array_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, hc, array_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(ha, d_a, array_bytes, cudaMemcpyDeviceToHost));
    t = mysecond();
#pragma omp parallel for
    for (j = 0; j < array_n; j++) 
        ha[j] = 2.0E0 * ha[j];
    
        CUDA_CHECK(cudaMemcpy(d_a, ha, array_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    t = 1.0E6 * (mysecond() - t);

    printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);

     /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar = 3.0;
    for (k=0; k<NTIMES; k++) 
    {
        times[0][k] = mysecond();
        CUDA_CHECK(cudaMemcpy(ha, d_a, array_bytes, cudaMemcpyDeviceToHost));
        kernel_copy(hc, ha, array_n);
        CUDA_CHECK(cudaMemcpy(d_c, hc, array_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        times[0][k] = mysecond() - times[0][k];

        times[1][k] = mysecond();
        CUDA_CHECK(cudaMemcpy(hc, d_c, array_bytes, cudaMemcpyDeviceToHost));
        kernel_scale(hb, hc, scalar, array_n);
        CUDA_CHECK(cudaMemcpy(d_b, hb, array_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        times[1][k] = mysecond() - times[1][k];

        times[2][k] = mysecond();
        CUDA_CHECK(cudaMemcpy(ha, d_a, array_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb, d_b, array_bytes, cudaMemcpyHostToDevice));
        kernel_add(hc, ha, hb, array_n);
        CUDA_CHECK(cudaMemcpy(d_c, hc, array_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        times[2][k] = mysecond() - times[2][k];

        times[3][k] = mysecond();
        CUDA_CHECK(cudaMemcpy(hb, d_b, array_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hc, d_c, array_bytes, cudaMemcpyDeviceToHost));
        kernel_triad(ha, hb, hc, scalar, array_n);
        CUDA_CHECK(cudaMemcpy(d_a, ha, array_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        times[3][k] = mysecond() - times[3][k];
    }

    /*	--- SUMMARY --- */

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
    {
    for (j=0; j<4; j++)
        {
        avgtime[j] = avgtime[j] + times[j][k];
        mintime[j] = MIN(mintime[j], times[j][k]);
        maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
    }

    printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
    for (j=0; j<4; j++) {
        avgtime[j] = avgtime[j]/(double)(NTIMES-1);

        printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j],
               1.0E-06 * bytes[j]/mintime[j],
               avgtime[j],
               mintime[j],
               maxtime[j]);
    }
    printf(HLINE);

    /* --- Check Results --- */
    checkSTREAMresults(ha, hb, hc, d_a, d_b, d_c, array_n, array_bytes);
    printf(HLINE);

    /* ---- Cleanup ---- */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(ha));
    CUDA_CHECK(cudaFreeHost(hb));
    CUDA_CHECK(cudaFreeHost(hc));

    return 0;
}

# define	M	20

int
checktick()
    {
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
	t1 = mysecond();
	while( ((t2=mysecond()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
	Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
    }



/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
void checkSTREAMresults(STREAM_TYPE *ha, STREAM_TYPE *hb,
                                STREAM_TYPE *hc,
                                STREAM_TYPE *d_a, STREAM_TYPE *d_b,
                                STREAM_TYPE *d_c, size_t n,
                                size_t array_bytes)
{
    STREAM_TYPE aj,bj,cj,scalar;
    STREAM_TYPE aSumErr,bSumErr,cSumErr;
    STREAM_TYPE aAvgErr,bAvgErr,cAvgErr;
    double epsilon;
    ssize_t	j;
	int	k,ierr,err;

    /* reproduce initialization */
    aj = 1.0; 
    bj = 2.0;
    cj = 0.0;
    /* a[] is modified during timing check */
    aj = 2.0E0 * aj;
    /* now execute timing loop */
    scalar = 3.0;
    for (k=0; k<NTIMES; k++) 
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }

     /* accumulate deltas between observed and expected results */
	aSumErr = 0.0;
	bSumErr = 0.0;
	cSumErr = 0.0;
    /* Pull final state from GPU */
    CUDA_CHECK(cudaMemcpy(ha, d_a, array_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb, d_b, array_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hc, d_c, array_bytes, cudaMemcpyDeviceToHost));

    for (j=0; j<n; j++) {
        aSumErr += abs(ha[j] - aj);
        bSumErr += abs(hb[j] - bj);
        cSumErr += abs(hc[j] - cj);
    }
    aAvgErr = aSumErr / (STREAM_TYPE)n;
    bAvgErr = bSumErr / (STREAM_TYPE)n;
    cAvgErr = cSumErr / (STREAM_TYPE)n;

    if (sizeof(STREAM_TYPE) == 4) {
		epsilon = 1.e-6;
	}
	else if (sizeof(STREAM_TYPE) == 8) {
		epsilon = 1.e-13;
	}
	else {
		printf("WEIRD: sizeof(STREAM_TYPE) = %lu\n",sizeof(STREAM_TYPE));
		epsilon = 1.e-6;
	}

    err = 0;
    if (abs(aAvgErr/aj) > epsilon) {
        err++;
        printf ("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,abs(aAvgErr)/aj);
        ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(ha[j]/aj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array a: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,aj,a[j],abs((aj-a[j])/aAvgErr));
				}
#endif
			}
		}
		printf("     For array a[], %d errors were found.\n",ierr);
    }
    if (abs(bAvgErr/bj) > epsilon) {
        err++;
        printf ("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,abs(bAvgErr)/bj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
        ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(hb[j]/bj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array b: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,bj,b[j],abs((bj-b[j])/bAvgErr));
				}
#endif
            }
        }
        printf("     For array b[], %d errors were found.\n",ierr);
    }
    if (abs(cAvgErr/cj) > epsilon) {
        err++;
        printf ("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,abs(cAvgErr)/cj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(hc[j]/cj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array c: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,cj,c[j],abs((cj-c[j])/cAvgErr));
				}
#endif
            }
        }
		printf("     For array c[], %d errors were found.\n",ierr);
    }
    if (err == 0) {
		printf ("Solution Validates: avg error less than %e on all three arrays\n",epsilon);
	}
    #ifdef VERBOSE
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif
}

void kernel_copy(STREAM_TYPE * __restrict__ c,
                         const STREAM_TYPE * __restrict__ a, size_t n)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t j = 0; j < n; j++) c[j] = a[j];
}

void kernel_scale(STREAM_TYPE * __restrict__ b,
                          const STREAM_TYPE * __restrict__ c,
                          STREAM_TYPE scalar, size_t n)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t j = 0; j < n; j++) b[j] = scalar * c[j];
}

void kernel_add(STREAM_TYPE * __restrict__ c,
                        const STREAM_TYPE * __restrict__ a,
                        const STREAM_TYPE * __restrict__ b, size_t n)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t j = 0; j < n; j++) c[j] = a[j] + b[j];
}

void kernel_triad(STREAM_TYPE * __restrict__ a,
                          const STREAM_TYPE * __restrict__ b,
                          const STREAM_TYPE * __restrict__ c,
                          STREAM_TYPE scalar, size_t n)
{
#ifdef _OPENMP
extern int omp_get_num_threads();
#endif
    for (size_t j = 0; j < n; j++) a[j] = b[j] + scalar * c[j];
}