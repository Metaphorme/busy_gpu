#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <thread>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <iostream>

static std::atomic<bool> g_stop{false};
static std::atomic<int>  g_sigcount{0};

// 打印当前时间的辅助函数
void print_current_time(const char* label) {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm local_tm = *std::localtime(&t);
    std::cout << "[" << label << "] "
              << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
}

// Ctrl+C 信号处理
void handle_sigint(int) {
    int c = ++g_sigcount;
    if (c == 1) {
        g_stop.store(true, std::memory_order_relaxed);
        fprintf(stderr, "\nCaught Ctrl+C -> stopping gracefully... (press Ctrl+C again to force exit)\n");
        print_current_time("End time");
    } else {
        fprintf(stderr, "Force exiting now.\n");
        _Exit(1);
    }
}

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { \
    cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)
#endif

__global__ void burn_kernel(float *sink, size_t offset, int iters_per_launch)
{
    const size_t gtid = offset + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    volatile float x = (float)(gtid & 0xFF) * 1.001f + 0.123f;
#pragma unroll 4
    for (int i = 0; i < iters_per_launch; ++i) {
        x = x * 1.0000001f + 1.0f;
        x = x * 0.9999999f + 0.5f;
        x = x * 1.0000002f + 0.25f;
        x = x * 0.9999998f + 0.125f;
    }
    sink[gtid] = x;
}

struct DeviceWorkerCfg {
    int device_id;
    int streams = 4;
    int threads_per_block = 256;
    int blocks_per_stream = 0;
    int iters_per_launch = 1<<18;
};

void device_worker(DeviceWorkerCfg cfg)
{
    CUDA_CHECK(cudaSetDevice(cfg.device_id));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device_id));

    const int sms = prop.multiProcessorCount;
    const int blocks_total = (cfg.blocks_per_stream > 0)
        ? cfg.blocks_per_stream * cfg.streams
        : sms * 32;
    const int blocks_per_stream = blocks_total / cfg.streams;

    const size_t threads_per_stream = (size_t)blocks_per_stream * cfg.threads_per_block;
    const size_t total_threads_all_streams = threads_per_stream * cfg.streams;

    float *d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, total_threads_all_streams * sizeof(float)));

    std::vector<cudaStream_t> ss(cfg.streams);
    std::vector<cudaEvent_t>  ev(cfg.streams);
    for (int i=0;i<cfg.streams;++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&ss[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming));
    }

    for (int i=0;i<cfg.streams;++i) {
        size_t offset = threads_per_stream * i;
        burn_kernel<<<blocks_per_stream, cfg.threads_per_block, 0, ss[i]>>>(d_sink, offset, 1<<10);
        CUDA_CHECK(cudaEventRecord(ev[i], ss[i]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Device %d: %s | SMs=%d | streams=%d | blocks/stream=%d | tpb=%d | iters/launch=%d\n",
        cfg.device_id, prop.name, sms, cfg.streams, blocks_per_stream, cfg.threads_per_block, cfg.iters_per_launch);

    while (!g_stop.load(std::memory_order_relaxed)) {
        for (int i=0;i<cfg.streams;++i) {
            if (cudaEventQuery(ev[i]) == cudaSuccess) {
                size_t offset = threads_per_stream * i;
                burn_kernel<<<blocks_per_stream, cfg.threads_per_block, 0, ss[i]>>>(d_sink, offset, cfg.iters_per_launch);
                CUDA_CHECK(cudaPeekAtLastError());
                CUDA_CHECK(cudaEventRecord(ev[i], ss[i]));
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    for (int i=0;i<cfg.streams;++i) CUDA_CHECK(cudaEventSynchronize(ev[i]));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i=0;i<cfg.streams;++i) { cudaEventDestroy(ev[i]); cudaStreamDestroy(ss[i]); }
    CUDA_CHECK(cudaFree(d_sink));
}

int main(int argc, char** argv)
{
    //

    printf("░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓███████▓▒░▒▓█▓▒░░▒▓█▓▒░       ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░ \n"
           "░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ \n"
           "░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ \n"
           "░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░ ░▒▓██████▓▒░       ░▒▓█▓▒▒▓███▓▒░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░ \n"
           "░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░  ░▒▓█▓▒░          ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ \n"
           "░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░  ░▒▓█▓▒░          ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ \n"
           "░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░   ░▒▓█▓▒░           ░▒▓██████▓▒░░▒▓█▓▒░       ░▒▓██████▓▒░  \n"
           "\n"
           "                                         Developer: Heqi Liu                                        \n"
           "\n");                                                                                       
                    
    //
    struct sigaction sa{};
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) { fprintf(stderr, "No CUDA device.\n"); return 1; }

    for (int i=0;i<ndev;++i)
        for (int j=0;j<ndev;++j)
            if (i!=j) {
                int access = 0;
                if (cudaDeviceCanAccessPeer(&access, i, j) == cudaSuccess && access) {
                    cudaSetDevice(i);
                    cudaDeviceEnablePeerAccess(j, 0);
                }
            }

    print_current_time("Start time");
    printf("Detected %d device(s). Burning all visible GPUs. Press Ctrl+C to stop (press twice to force).\n", ndev);

    std::vector<std::thread> workers;
    workers.reserve(ndev);

    const char* ev_streams = std::getenv("BURN_STREAMS");
    const char* ev_tpb     = std::getenv("BURN_TPB");
    const char* ev_iters   = std::getenv("BURN_ITERS");

    for (int d=0; d<ndev; ++d) {
        DeviceWorkerCfg cfg;
        cfg.device_id = d;
        if (ev_streams) cfg.streams = std::max(1, atoi(ev_streams));
        if (ev_tpb)     cfg.threads_per_block = std::max(32, atoi(ev_tpb));
        if (ev_iters)   cfg.iters_per_launch  = std::max(1024, atoi(ev_iters));
        workers.emplace_back(device_worker, cfg);
    }

    for (auto& t : workers) t.join();
    printf("Stopped.\n");
    return 0;
}
