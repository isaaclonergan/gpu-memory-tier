#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cuda_runtime.h>

#ifndef DRAM_MiB
#   define DRAM_MiB  256
#endif

#ifndef VRAM_ACCESSES
#   define VRAM_ACCESSES  5000
#endif

#ifndef NVME_ACCESSES
#   define NVME_ACCESSES  5000
#endif

#define NVME_READ_SIZE  4096

#define HLINE "-------------------------------------------------------------\n"
#define GiB (1024ULL*1024ULL*1024ULL)
#define MiB (1024ULL*1024ULL)

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while(0)

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

static void shuffle(size_t *arr, size_t n)
{
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)rand() % (i + 1);
        size_t tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}


static double measure_dram_latency(void)
{
    size_t n = ((size_t)DRAM_MiB * MiB) / sizeof(size_t);

    size_t *arr = (size_t *)malloc(n * sizeof(size_t));
    if (!arr) {
        fprintf(stderr, "Failed to allocate %d MiB DRAM buffer\n", DRAM_MiB);
        exit(EXIT_FAILURE);
    }

    size_t *perm = (size_t *)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) perm[i] = i;
    shuffle(perm, n);

    for (size_t i = 0; i < n - 1; i++)
        arr[perm[i]] = perm[i + 1];
    arr[perm[n - 1]] = perm[0];
    free(perm);

    volatile size_t idx = 0;
    for (size_t i = 0; i < n; i++)
        idx = arr[idx];

    double t0 = mysecond();
    idx = 0;
    for (size_t i = 0; i < n; i++)
        idx = arr[idx];
    double t1 = mysecond();

    if (idx == 0) arr[0] = 0;

    double ns_per_access = (t1 - t0) * 1.0e9 / (double)n;
    free(arr);
    return ns_per_access;
}

static double measure_vram_latency(void)
{
    size_t vram_mib = 64;
    size_t n        = (vram_mib * MiB) / sizeof(size_t);

    size_t *h_arr = (size_t *)malloc(n * sizeof(size_t));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host buffer for VRAM chase\n");
        exit(EXIT_FAILURE);
    }
    size_t *perm = (size_t *)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) perm[i] = i;
    shuffle(perm, n);
    for (size_t i = 0; i < n - 1; i++)
        h_arr[perm[i]] = perm[i + 1];
    h_arr[perm[n - 1]] = perm[0];
    free(perm);

    size_t *d_arr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_arr, n * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, n * sizeof(size_t),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    free(h_arr);

    size_t idx = 0;
    size_t next;
    for (int w = 0; w < 100; w++) {
        CUDA_CHECK(cudaMemcpy(&next, d_arr + idx, sizeof(size_t),
                               cudaMemcpyDeviceToHost));
        idx = next % n;
    }

    idx = 0;
    double t0 = mysecond();
    for (int i = 0; i < VRAM_ACCESSES; i++) {
        CUDA_CHECK(cudaMemcpy(&next, d_arr + idx, sizeof(size_t),
                               cudaMemcpyDeviceToHost));
        idx = next % n;
    }
    double t1 = mysecond();

    double us_per_access = (t1 - t0) * 1.0e6 / (double)VRAM_ACCESSES;
    CUDA_CHECK(cudaFree(d_arr));
    return us_per_access;
}

static double measure_nvme_latency(void)
{
    size_t file_bytes = (size_t)2 * GiB;
    size_t n_blocks   = file_bytes / NVME_READ_SIZE;

    printf("  Creating 2 GiB latency probe file (latbench_probe.bin)...\n");
    fflush(stdout);

    {
        int fd = open("latbench_probe.bin", O_WRONLY|O_CREAT|O_TRUNC, 0644);
        if (fd < 0) {
            fprintf(stderr, "open latbench_probe.bin: %s\n", strerror(errno));
            exit(EXIT_FAILURE);
        }
        size_t chunk = 64 * MiB;
        uint8_t *buf = (uint8_t *)malloc(chunk);
        for (size_t i = 0; i < chunk; i++) buf[i] = (uint8_t)(i ^ 0xA5);
        size_t written = 0;
        while (written < file_bytes) {
            size_t to_write = (file_bytes - written < chunk) ?
                               file_bytes - written : chunk;
            ssize_t n = write(fd, buf, to_write);
            if (n < 0) {
                fprintf(stderr, "write: %s\n", strerror(errno));
                exit(EXIT_FAILURE);
            }
            written += (size_t)n;
        }
        free(buf);
        fsync(fd);
        close(fd);
    }

    /* Drop page cache */
    sync();
    {
        int fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
        if (fd >= 0) { write(fd, "3\n", 2); close(fd); }
        else printf("  Note: could not drop cache (run as root for best results)\n");
    }

    /* Open with O_DIRECT for reads */
    int fd = open("latbench_probe.bin", O_RDONLY | O_DIRECT);
    if (fd < 0) {
        fprintf(stderr, "open latbench_probe.bin O_DIRECT: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Aligned read buffer */
    uint8_t *buf = (uint8_t *)aligned_alloc(4096, NVME_READ_SIZE);
    if (!buf) { fprintf(stderr, "aligned_alloc failed\n"); exit(EXIT_FAILURE); }

    /* Warmup */
    for (int w = 0; w < 32; w++) {
        size_t blk = (size_t)rand() % n_blocks;
        pread(fd, buf, NVME_READ_SIZE, (off_t)(blk * NVME_READ_SIZE));
    }

    /* Drop cache again after warmup */
    sync();
    {
        int cfd = open("/proc/sys/vm/drop_caches", O_WRONLY);
        if (cfd >= 0) { write(cfd, "3\n", 2); close(cfd); }
    }

    size_t blk = 0;
    double t0 = mysecond();
    for (int i = 0; i < NVME_ACCESSES; i++) {
        off_t offset = (off_t)((blk % n_blocks) * NVME_READ_SIZE);
        ssize_t r = pread(fd, buf, NVME_READ_SIZE, offset);
        if (r < 0) {
            fprintf(stderr, "pread: %s\n", strerror(errno));
            exit(EXIT_FAILURE);
        }
        uint64_t val;
        memcpy(&val, buf, sizeof(uint64_t));
        blk = (blk * 6364136223846793005ULL + val) % n_blocks;
    }
    double t1 = mysecond();

    double us_per_access = (t1 - t0) * 1.0e6 / (double)NVME_ACCESSES;
    free(buf);
    close(fd);
    return us_per_access;
}

int main(void)
{
    int deviceId = 0;
    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    CUDA_CHECK(cudaSetDevice(deviceId));

    srand(42);

    printf(HLINE);
    printf("LATBENCH -- Memory hierarchy latency measurement\n");
    printf(HLINE);
    printf("GPU      : %s\n", prop.name);
    printf("DRAM     : pointer-chase over %d MiB (exceeds L3 cache)\n", DRAM_MiB);
    printf("VRAM     : pointer-chase over 64 MiB in GPU VRAM, %d steps\n",
           VRAM_ACCESSES);
    printf("NVMe     : random O_DIRECT pread (%d byte blocks), %d accesses\n",
           NVME_READ_SIZE, NVME_ACCESSES);
    printf(HLINE);
    printf("\n");

    /* ---- DRAM ---- */
    printf("Measuring DRAM latency (pointer chase, %d MiB)...\n", DRAM_MiB);
    fflush(stdout);
    double dram_ns = measure_dram_latency();
    printf("  DRAM latency : %.1f ns\n\n", dram_ns);

    /* ---- VRAM ---- */
    printf("Measuring VRAM latency (%d serialized cudaMemcpy reads)...\n",
           VRAM_ACCESSES);
    fflush(stdout);
    double vram_us = measure_vram_latency();
    printf("  VRAM latency : %.2f us  (%.0f ns)\n\n",
           vram_us, vram_us * 1000.0);

    /* ---- NVMe ---- */
    printf("Measuring NVMe latency (%d random O_DIRECT reads)...\n",
           NVME_ACCESSES);
    fflush(stdout);
    double nvme_us = measure_nvme_latency();
    printf("  NVMe latency : %.1f us\n\n", nvme_us);

    /* ---- Summary ---- */
    printf(HLINE);
    printf("LATENCY SUMMARY\n");
    printf(HLINE);
    printf("  %-20s  %12s  %12s  %s\n", "Level", "Latency", "", "Relative to DRAM");
    printf("  %-20s  %12s  %12s  %s\n", "-----", "-------", "", "----------------");
    printf("  %-20s  %9.1f ns  %12s  1x\n",
           "DRAM", dram_ns, "");
    printf("  %-20s  %9.0f ns  %9.2f us  %.0fx slower than DRAM\n",
           "GPU VRAM (PCIe)", vram_us * 1000.0, vram_us,
           (vram_us * 1000.0) / dram_ns);
    printf("  %-20s  %9.0f ns  %9.1f us  %.0fx slower than DRAM\n",
           "NVMe", nvme_us * 1000.0, nvme_us,
           (nvme_us * 1000.0) / dram_ns);
    printf(HLINE);
    printf("\n");
    printf("Note: VRAM latency = full PCIe round-trip (CPU issues read,\n");
    printf("      travels to GPU, GPU reads GDDR6, result returns to CPU).\n");
    printf("      Not the same as GPU-internal GDDR6 latency (~100ns).\n");
    printf("\n");
    printf("Files left on disk: latbench_probe.bin (delete when done)\n");
    printf(HLINE);

    return 0;
}
