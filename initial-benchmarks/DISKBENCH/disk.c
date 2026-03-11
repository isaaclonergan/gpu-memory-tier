/* 
    Disk Benchmark
    Heavily inspired, and most code copied, from STREAM by John D. McCalpin
*/

#define _GNU_SOURCE

# include <stdio.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
#include <stdlib.h>
#include <fcntl.h>

#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif

/*  2) STREAM runs each kernel "NTIMES" times and reports the *best* result
 *         for any iteration after the first, therefore the minimum value
 *         for NTIMES is 2.
 *      There are no rules on maximum allowable values for NTIMES, but
 *         values larger than the default are unlikely to noticeably
 *         increase the reported performance.
 *      NTIMES can also be set on the compile line without changing the source
 *         code using, for example, "-DNTIMES=7".
 */
#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif

#define BLOCK_SIZE 4096

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
extern void checkSTREAMresults(STREAM_TYPE*,STREAM_TYPE*,STREAM_TYPE*);
#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

static void drop_cache();
static void read_array(const char*,STREAM_TYPE*,size_t);
static void write_array(const char*,STREAM_TYPE*,size_t);
static size_t block_align_up(size_t);

int
main()
    {
    int			quantum, checktick();
    int			BytesPerWord;
    int			k;
    ssize_t		j;
    STREAM_TYPE		scalar;
    double		t, times[4][NTIMES];

    size_t array_bytes = STREAM_ARRAY_SIZE*sizeof(STREAM_TYPE);
    size_t file_bytes  = block_align_up(array_bytes);
    size_t buf_n       = file_bytes/sizeof(STREAM_TYPE);

    STREAM_TYPE *a   = aligned_alloc(BLOCK_SIZE,file_bytes);
    STREAM_TYPE *b   = aligned_alloc(BLOCK_SIZE,file_bytes);
    STREAM_TYPE *c   = aligned_alloc(BLOCK_SIZE,file_bytes);
    STREAM_TYPE *tmp = aligned_alloc(BLOCK_SIZE,file_bytes);

    if (!a||!b||!c||!tmp) {
        printf("Allocation failed\n");
        exit(1);
    }

    printf(HLINE);
    printf("STREAM Disk Variant\n");
    printf(HLINE);
    BytesPerWord = sizeof(STREAM_TYPE);
    printf("This system uses %d bytes per array element.\n",
	BytesPerWord);

    printf("Array size = %llu (elements)\n" , (unsigned long long) STREAM_ARRAY_SIZE);
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

#ifdef _OPENMP
	k = 0;
#pragma omp parallel
#pragma omp atomic 
		k++;
    printf ("Number of Threads counted = %i\n",k);
#endif

    /* Get initial value for system clock. */
#pragma omp parallel for
    for (j=0; j<buf_n; j++) {

        if (j<STREAM_ARRAY_SIZE) {
            a[j]=1.0;
            b[j]=2.0;
            c[j]=0.0;
        }
        else {
            a[j]=0;
            b[j]=0;
            c[j]=0;
        }

        tmp[j]=0;
    }

    write_array("stream_a.bin",a,file_bytes);
    write_array("stream_b.bin",b,file_bytes);
    write_array("stream_c.bin",c,file_bytes);

    read_array("stream_a.bin",a,file_bytes);

#pragma omp parallel for
    for (j = 0; j < STREAM_ARRAY_SIZE; j++)
        a[j] = 2.0E0 * a[j];

    write_array("stream_a.bin",a,file_bytes);

    quantum = checktick();

    printf("Clock granularity %d microseconds\n",quantum);

    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
    {
        times[0][k] = mysecond();

        read_array("stream_a.bin",a,file_bytes);

#pragma omp parallel for
        for (j=0;j<STREAM_ARRAY_SIZE;j++)
            tmp[j]=a[j];

        write_array("stream_c.bin",tmp,file_bytes);

        times[0][k] = mysecond() - times[0][k];

        times[1][k] = mysecond();

        read_array("stream_c.bin",c,file_bytes);

#pragma omp parallel for
        for (j=0;j<STREAM_ARRAY_SIZE;j++)
            tmp[j]=scalar*c[j];

        write_array("stream_b.bin",tmp,file_bytes);

        times[1][k] = mysecond() - times[1][k];

        times[2][k] = mysecond();

        read_array("stream_a.bin",a,file_bytes);
        read_array("stream_b.bin",b,file_bytes);

#pragma omp parallel for
        for (j=0; j<STREAM_ARRAY_SIZE; j++)
            tmp[j] = a[j]+b[j];

        write_array("stream_c.bin",tmp,file_bytes);

        times[2][k] = mysecond() - times[2][k];

        times[3][k] = mysecond();

        read_array("stream_b.bin",b,file_bytes);
        read_array("stream_c.bin",c,file_bytes);

#pragma omp parallel for
        for (j=0; j<STREAM_ARRAY_SIZE; j++)
            tmp[j] = b[j]+scalar*c[j];

        write_array("stream_a.bin",tmp,file_bytes);

        times[3][k] = mysecond() - times[3][k];
    }


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

    read_array("stream_a.bin",a,file_bytes);
    read_array("stream_b.bin",b,file_bytes);
    read_array("stream_c.bin",c,file_bytes);

    checkSTREAMresults(a,b,c);

    printf(HLINE);

    free(a);
    free(b);
    free(c);
    free(tmp);

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
void checkSTREAMresults(STREAM_TYPE *a,STREAM_TYPE *b,STREAM_TYPE *c)
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
	for (j=0; j<STREAM_ARRAY_SIZE; j++) {
		aSumErr += abs(a[j] - aj);
		bSumErr += abs(b[j] - bj);
		cSumErr += abs(c[j] - cj);
		// if (j == 417) printf("Index 417: c[j]: %f, cj: %f\n",c[j],cj);	// MCCALPIN
	}
	aAvgErr = aSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	bAvgErr = bSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	cAvgErr = cSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;

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
			if (abs(a[j]/aj-1.0) > epsilon) {
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
			if (abs(b[j]/bj-1.0) > epsilon) {
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
			if (abs(c[j]/cj-1.0) > epsilon) {
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

static size_t block_align_up(size_t n)
{
    return (n + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
}

static int can_drop_cache = 0;

static void drop_cache()
{
    if (!can_drop_cache) return;

    sync();

    int fd=open("/proc/sys/vm/drop_caches",O_WRONLY);
    if (fd<0) return;

    write(fd,"3\n",2);
    close(fd);
}

static void read_array(const char *path,STREAM_TYPE *buf,size_t bytes)
{
    drop_cache();

    int fd=open(path,O_RDONLY|O_DIRECT);
    if (fd<0) {
        perror("open");
        exit(1);
    }

    size_t done=0;
    char *p=(char*)buf;

    while(done<bytes) {
        ssize_t n=read(fd,p+done,bytes-done);
        if(n<=0) exit(1);
        done+=n;
    }

    close(fd);
}

static void write_array(const char *path,STREAM_TYPE *buf,size_t bytes)
{
    int fd=open(path,O_WRONLY|O_CREAT|O_DIRECT,0644);

    if(fd<0)
        fd=open(path,O_WRONLY|O_CREAT,0644);

    if(fd<0) exit(1);

    size_t done=0;
    char *p=(char*)buf;

    while(done<bytes) {
        ssize_t n=write(fd,p+done,bytes-done);
        if(n<=0) exit(1);
        done+=n;
    }

    fsync(fd);
    close(fd);
}