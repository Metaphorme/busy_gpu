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
#include <cstring>

// Version info from Makefile
#ifndef VERSION
#define VERSION "unknown"
#endif
#ifndef BUILD_DATE
#define BUILD_DATE "unknown"
#endif
#ifndef BUILD_OS
#define BUILD_OS "unknown"
#endif
#ifndef BUILD_ARCH
#define BUILD_ARCH "unknown"
#endif
#ifndef GPU_ARCHS
#define GPU_ARCHS "unknown"
#endif

static std::atomic<bool> g_stop{false};
static std::atomic<int>  g_sigcount{0};
static std::atomic<bool> g_stats_printed{false}; // 标记是否已打印统计信息
static std::chrono::system_clock::time_point g_start_time;
static std::atomic<long long>* g_device_kernel_launches = nullptr;
static int g_total_devices = 0;
static long long g_target_duration_sec = -1; // -1表示无限运行

// 打印当前时间的辅助函数
void print_current_time(const char* label) {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm local_tm = *std::localtime(&t);
    std::cout << "[" << label << "] "
              << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
}

// 格式化持续时间
std::string format_duration(long long seconds) {
    long long days = seconds / 86400;
    long long hours = (seconds % 86400) / 3600;
    long long mins = (seconds % 3600) / 60;
    long long secs = seconds % 60;
    
    char buf[128];
    if (days > 0) {
        snprintf(buf, sizeof(buf), "%lld days %lld hours %lld minutes %lld seconds", days, hours, mins, secs);
    } else if (hours > 0) {
        snprintf(buf, sizeof(buf), "%lld hours %lld minutes %lld seconds", hours, mins, secs);
    } else if (mins > 0) {
        snprintf(buf, sizeof(buf), "%lld minutes %lld seconds", mins, secs);
    } else {
        snprintf(buf, sizeof(buf), "%lld seconds", secs);
    }
    return std::string(buf);
}

// 解析时间参数（如 1s, 2m, 3h, 4d）
long long parse_duration(const char* str) {
    if (!str || strlen(str) < 2) return -1;
    
    char* endptr;
    long long value = strtoll(str, &endptr, 10);
    if (value <= 0) return -1;
    
    char unit = *endptr;
    switch (unit) {
        case 's': case 'S': return value;
        case 'm': case 'M': return value * 60;
        case 'h': case 'H': return value * 3600;
        case 'd': case 'D': return value * 86400;
        default: return -1;
    }
}

// 打印版本信息
void print_version() {
    printf("Busy GPU v%s\n", VERSION);
    printf("Build date: %s\n", BUILD_DATE);
    printf("Build OS: %s\n", BUILD_OS);
    printf("Build architecture: %s\n", BUILD_ARCH);
    printf("Supported CUDA SM versions: %s\n", GPU_ARCHS);
    
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    printf("CUDA Runtime version: %d.%d\n", runtime_version / 1000, (runtime_version % 100) / 10);
    
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    printf("CUDA Driver version: %d.%d\n", driver_version / 1000, (driver_version % 100) / 10);
}

// 打印帮助信息
void print_help(const char* prog_name) {
    printf("Usage: %s [OPTIONS]\n\n", prog_name);
    printf("A simple GPU stress test tool for Nvidia cards.\n\n");
    printf("OPTIONS:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -v, -V, --version       Show version and build information\n");
    printf("  -t, --time DURATION     Run for specified duration (e.g., 10s, 5m, 2h, 1d)\n");
    printf("                          s=seconds, m=minutes, h=hours, d=days\n");
    printf("  -d, --devices IDs       Specify GPU devices to use (e.g., 0,1,2)\n");
    printf("                          Default: all visible devices\n");
    printf("\n");
    printf("ENVIRONMENT VARIABLES:\n");
    printf("  BURN_STREAMS            Number of CUDA streams per device (default: 4)\n");
    printf("  BURN_TPB                Threads per block (default: 256)\n");
    printf("  BURN_ITERS              Iterations per kernel launch (default: 262144)\n");
    printf("  CUDA_VISIBLE_DEVICES    Limit visible GPUs to CUDA\n");
    printf("\n");
    printf("EXAMPLES:\n");
    printf("  %s                      Run indefinitely on all GPUs\n", prog_name);
    printf("  %s -t 30s               Run for 30 seconds\n", prog_name);
    printf("  %s -t 2h                Run for 2 hours\n", prog_name);
    printf("  %s -d 0,2               Run on GPU 0 and 2 only\n", prog_name);
    printf("  CUDA_VISIBLE_DEVICES=0,1 %s  Run on GPU 0 and 1\n", prog_name);
    printf("\n");
    printf("SIGNALS:\n");
    printf("  Ctrl+C (SIGINT)         Graceful shutdown with statistics\n");
    printf("  SIGTERM, SIGHUP         Graceful shutdown\n");
    printf("  Send signal twice       Force immediate exit\n");
    printf("\n");
}

// 打印统计信息和评分
void print_statistics_and_scores() {
    // 如果已经打印过，就不再打印
    bool expected = false;
    if (!g_stats_printed.compare_exchange_strong(expected, true)) {
        return;
    }
    
    auto end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - g_start_time);
    double duration_sec = duration.count();
    
    print_current_time("End time");
    std::cout << "Duration: " << format_duration(duration.count()) << " (" << duration.count() << " seconds)" << std::endl;
    
    if (g_device_kernel_launches && duration_sec > 0) {
        std::cout << "\n========== Performance Scores ==========" << std::endl;
        double total_score = 0.0;
        long long total_launches = 0;
        
        for (int i = 0; i < g_total_devices; ++i) {
            long long launches = g_device_kernel_launches[i].load(std::memory_order_relaxed);
            total_launches += launches;
            // 单卡评分：每秒的kernel启动次数作为基础分数
            double device_score = (double)launches / duration_sec;
            total_score += device_score;
            
            printf("Device %d: %lld kernel launches, %.2f launches/sec, Score: %.2f\n", 
                   i, launches, device_score, device_score);
        }
        
        printf("\nTotal launches: %lld\n", total_launches);
        printf("Total Score: %.2f\n", total_score);
        printf("Average Score per GPU: %.2f\n", total_score / g_total_devices);
        std::cout << "========================================" << std::endl;
    }
}

// 信号处理（支持多种退出信号）
void handle_signal(int signum) {
    int c = ++g_sigcount;
    if (c == 1) {
        g_stop.store(true, std::memory_order_relaxed);
        const char* signame = "UNKNOWN";
        switch(signum) {
            case SIGINT:  signame = "SIGINT (Ctrl+C)"; break;
            case SIGTERM: signame = "SIGTERM"; break;
            case SIGHUP:  signame = "SIGHUP"; break;
            case SIGQUIT: signame = "SIGQUIT"; break;
            default: break;
        }
        fprintf(stderr, "\nCaught signal %s -> stopping gracefully... (send again to force exit)\n", signame);
        print_statistics_and_scores();
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
    std::atomic<long long>* kernel_count = nullptr;
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
        // 检查是否超时
        if (g_target_duration_sec > 0) {
            auto now = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time);
            if (elapsed.count() >= g_target_duration_sec) {
                g_stop.store(true, std::memory_order_relaxed);
                break;
            }
        }
        
        for (int i=0;i<cfg.streams;++i) {
            if (cudaEventQuery(ev[i]) == cudaSuccess) {
                size_t offset = threads_per_stream * i;
                burn_kernel<<<blocks_per_stream, cfg.threads_per_block, 0, ss[i]>>>(d_sink, offset, cfg.iters_per_launch);
                CUDA_CHECK(cudaPeekAtLastError());
                CUDA_CHECK(cudaEventRecord(ev[i], ss[i]));
                // 统计kernel启动次数
                if (cfg.kernel_count) {
                    cfg.kernel_count->fetch_add(1, std::memory_order_relaxed);
                }
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
    // 解析命令行参数
    std::vector<int> selected_devices;
    bool show_help = false;
    bool show_version = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg == "-v" || arg == "-V" || arg == "--version") {
            show_version = true;
        } else if (arg == "-t" || arg == "--time") {
            if (i + 1 < argc) {
                g_target_duration_sec = parse_duration(argv[++i]);
                if (g_target_duration_sec < 0) {
                    fprintf(stderr, "Error: Invalid duration format '%s'. Use format like: 10s, 5m, 2h, 1d\n", argv[i]);
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: -t/--time requires a duration argument\n");
                return 1;
            }
        } else if (arg == "-d" || arg == "--devices") {
            if (i + 1 < argc) {
                char* devices_str = argv[++i];
                char* token = strtok(devices_str, ",");
                while (token) {
                    int dev_id = atoi(token);
                    selected_devices.push_back(dev_id);
                    token = strtok(nullptr, ",");
                }
            } else {
                fprintf(stderr, "Error: -d/--devices requires device IDs argument\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Error: Unknown option '%s'\n", arg.c_str());
            fprintf(stderr, "Use -h or --help for usage information\n");
            return 1;
        }
    }
    
    // 处理 --help
    if (show_help) {
        print_help(argv[0]);
        return 0;
    }
    
    // 处理 --version
    if (show_version) {
        print_version();
        return 0;
    }

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
           "                                           Version: %s                                              \n"
           "                         Repository: https://github.com/Metaphorme/busy_gpu                         \n"
           "                           A simple GPU stress test tool for Nvidia cards.                          \n"
           "\n", VERSION);                                                                                       
                    
    //
    struct sigaction sa{};
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    // 注册多种信号处理
    sigaction(SIGINT, &sa, nullptr);   // Ctrl+C
    sigaction(SIGTERM, &sa, nullptr);  // kill 命令
    sigaction(SIGHUP, &sa, nullptr);   // 终端关闭
    sigaction(SIGQUIT, &sa, nullptr);  // Ctrl+\

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) { fprintf(stderr, "No CUDA device.\n"); return 1; }
    
    // 验证选定的设备ID
    if (!selected_devices.empty()) {
        for (int dev_id : selected_devices) {
            if (dev_id < 0 || dev_id >= ndev) {
                fprintf(stderr, "Error: Invalid device ID %d. Available devices: 0-%d\n", dev_id, ndev-1);
                return 1;
            }
        }
    } else {
        // 如果没有指定设备，使用所有设备
        for (int i = 0; i < ndev; ++i) {
            selected_devices.push_back(i);
        }
    }

    // 初始化全局统计变量
    g_total_devices = selected_devices.size();
    g_device_kernel_launches = new std::atomic<long long>[g_total_devices];
    for (int i = 0; i < g_total_devices; ++i) {
        g_device_kernel_launches[i].store(0, std::memory_order_relaxed);
    }

    // 注册atexit清理函数，确保正常退出时也打印统计
    std::atexit([]() {
        if (!g_stop.load(std::memory_order_relaxed)) {
            print_statistics_and_scores();
        }
        if (g_device_kernel_launches) {
            delete[] g_device_kernel_launches;
            g_device_kernel_launches = nullptr;
        }
    });

    for (int i : selected_devices)
        for (int j : selected_devices)
            if (i!=j) {
                int access = 0;
                if (cudaDeviceCanAccessPeer(&access, i, j) == cudaSuccess && access) {
                    cudaSetDevice(i);
                    cudaDeviceEnablePeerAccess(j, 0);
                }
            }

    g_start_time = std::chrono::system_clock::now();
    print_current_time("Start time");
    
    if (g_target_duration_sec > 0) {
        printf("Detected %d device(s), using %d device(s). Running for %s.\n", 
               ndev, g_total_devices, format_duration(g_target_duration_sec).c_str());
    } else {
        printf("Detected %d device(s), using %d device(s). Running indefinitely. Press Ctrl+C to stop.\n", 
               ndev, g_total_devices);
    }
    
    if (g_total_devices < ndev) {
        printf("Selected devices: ");
        for (size_t i = 0; i < selected_devices.size(); ++i) {
            printf("%d%s", selected_devices[i], i < selected_devices.size() - 1 ? ", " : "\n");
        }
    }
    printf("\n");

    std::vector<std::thread> workers;
    workers.reserve(ndev);

    const char* ev_streams = std::getenv("BURN_STREAMS");
    const char* ev_tpb     = std::getenv("BURN_TPB");
    const char* ev_iters   = std::getenv("BURN_ITERS");

    for (size_t idx = 0; idx < selected_devices.size(); ++idx) {
        int d = selected_devices[idx];
        DeviceWorkerCfg cfg;
        cfg.device_id = d;
        cfg.kernel_count = &g_device_kernel_launches[idx];
        if (ev_streams) cfg.streams = std::max(1, atoi(ev_streams));
        if (ev_tpb)     cfg.threads_per_block = std::max(32, atoi(ev_tpb));
        if (ev_iters)   cfg.iters_per_launch  = std::max(1024, atoi(ev_iters));
        workers.emplace_back(device_worker, cfg);
    }

    for (auto& t : workers) t.join();
    
    // 正常退出时打印统计信息
    if (g_stop.load(std::memory_order_relaxed)) {
        // 先打印退出原因
        if (g_target_duration_sec > 0 && !g_stats_printed.load(std::memory_order_relaxed)) {
            printf("\nTime limit reached.\n");
        } else if (!g_stats_printed.load(std::memory_order_relaxed)) {
            printf("\nBye!\n");
        }
        // 然后打印统计（如果还没打印过）
        print_statistics_and_scores();
    }
    
    return 0;
}
