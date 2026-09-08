// Standalone CUDA init probe — isolates "ggml_cuda_init: failed to initialize CUDA"
// from llama's logging so you get the exact error code + string.
//   nvcc -o /tmp/cuda-probe scripts/cuda-probe.cu && /tmp/cuda-probe
// Reads nothing, needs no network. Interpret the output per SERVE-GLM-NOTES.md.
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);   // CUDA version this binary was built against
    cudaDriverGetVersion(&drv);   // max CUDA version the INSTALLED DRIVER supports
    printf("runtime_built_against = %d.%d\n", rt / 1000, (rt % 1000) / 10);
    printf("driver_supports_up_to = %d.%d\n", drv / 1000, (drv % 1000) / 10);
    if (drv < rt) {
        printf(">> DRIVER TOO OLD for this build (would give 'insufficient driver')\n");
    }

    int n = -1;
    cudaError_t e = cudaGetDeviceCount(&n);
    printf("cudaGetDeviceCount -> err=%d (%s), count=%d\n",
           (int) e, cudaGetErrorString(e), n);

    if (e == cudaSuccess) {
        for (int i = 0; i < n; ++i) {
            cudaDeviceProp p;
            if (cudaGetDeviceProperties(&p, i) == cudaSuccess) {
                printf("  dev %d: %-28s sm_%d%d  %zu MiB\n",
                       i, p.name, p.major, p.minor, p.totalGlobalMem / (1024 * 1024));
            }
        }
    }
    return (e == cudaSuccess) ? 0 : 1;
}
