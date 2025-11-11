# Busy GPU

English document | [中文文档](./README_zh.md)

A GPU stress testing and burn-in tool for testing GPU stability and performance.

## Compilation

Using Makefile (recommended):

```bash
make          # Build
make clean    # Clean
make rebuild  # Rebuild
make install  # Install to system (requires sudo)
``````

Or compile manually:

```bash
nvcc -O3 -use_fast_math busy_gpu.cu -o busy_gpu

# Specify SM versions
nvcc -O3 -use_fast_math busy_gpu.cu -o busy_gpu \
  -cudart static \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89
```

## Usage

### Basic Usage

```bash
# Run indefinitely on all GPUs
./busy_gpu

# Show help
./busy_gpu --help./busy_gpu --help

# Show version information
./busy_gpu --version./busy_gpu --version

# Specify running time

./busy_gpu -t 30s    # Run for 30 seconds
./busy_gpu -t 5m     # Run for 5 minutes
./busy_gpu -t 2h     # Run for 2 hours
./busy_gpu -t 1d     # Run for 1 day

# Specify GPU devices to use
./busy_gpu -d 0,2    # Use GPU 0 and 2 only

# Combined usage
./busy_gpu -t 1h -d 0,1    # Run on GPU 0 and 1 for 1 hour
``````

### Environment Variables

Adjust program behavior through environment variables:

```bash
# Set number of CUDA streams per device (default: 4)
BURN_STREAMS=8 ./busy_gpu

# Set threads per block (default: 256)
BURN_TPB=512 ./busy_gpu

# Set iterations per kernel launch (default: 262144)
BURN_ITERS=524288 ./busy_gpu

# Limit visible GPUs

CUDA_VISIBLE_DEVICES=0,1 ./busy_gpu
``````

### Using timeout command

```bash
# Run for 12 hours using timeout command
timeout 12h ./busy_gpu
``````

## Features

- ✅ Multi-GPU parallel stress testing

- ✅ Performance scoring system

- ✅ Timed run support

- ✅ Flexible device selection

- ✅ Graceful signal handling

- ✅ Detailed statistics output



## How It Works

### Stress Testing Mechanism

Busy GPU applies computational pressure to the GPU by continuously launching CUDA kernels. The core mechanism is as follows:

1. **Multi-Stream Concurrent Execution**

   - Creates multiple CUDA streams per GPU (default: 4)

   - Multiple streams execute concurrently, maximizing GPU parallelism

   - Uses events to detect kernel completion status

2. **Compute-Intensive Kernel**

    ```cuda
   // Each thread executes massive floating-point operations
   for (int i = 0; i < iters_per_launch; ++i) {
       x = x * 1.0000001f + 1.0f;   // Floating-point multiply-add
       x = x * 0.9999999f + 0.5f;
       x = x * 1.0000002f + 0.25f;
       x = x * 0.9999998f + 0.125f;
   }
   ```

   - Each kernel launch executes 262,144 iterations (configurable)

   - Each iteration contains 4 sets of floating-point multiply-add operations

   - Total of over 1 million floating-point operations per thread

3. **Resource Allocation Strategy**

   - Dynamically adjusts block count based on GPU's SM (Streaming Multiprocessor) count

   - Default allocation of 32 blocks per SM ensures full hardware utilization

   - Each block contains 256 threads (configurable)

4. **Continuous Work Loop**

   ```

   Initialize → Warm-up Launch → Loop Detection

                                      ↓

            ← Launch New Kernel ← Event Complete?

   ```

   - Uses non-blocking event queries

   - Launches new kernel immediately upon completion

   - Keeps GPU at full utilization



### Why It's Effective for Testing GPUs

1. **Comprehensive Hardware Coverage**

   - Simultaneously uses all SM units

   - Massive floating-point operations exercise ALUs (Arithmetic Logic Units)

   - Frequent memory accesses test VRAM and memory controllers



2. **Sustained High Power Consumption**

   - Intensive computation keeps GPU cores at high frequency

   - Generates substantial heat, testing cooling system

   - Long-term operation reveals temperature-related stability issues



3. **Reproducible Workload**

   - Computation is deterministic with same configuration

   - Performance scores enable system comparisons

   - Score drops may indicate hardware issues or thermal throttling

## Signal Handling

The program supports the following signals:

- `Ctrl+C` (SIGINT): Graceful shutdown with statistics

- `SIGTERM`: Graceful shutdown

- `SIGHUP`: Graceful shutdown

- `SIGQUIT`: Graceful shutdown

Send signal twice to force immediate exit.

## Performance Metrics

The program outputs detailed performance statistics at completion, including:

### Per-Device Metrics

- **kernel launches**: Total kernel launches on this GPU during the run

- **launches/sec**: Average kernels launched per second (kernel launches ÷ duration)

- **Score**: Per-device performance score, equals launches/sec

### Overall Metrics

- **Total launches**: Sum of kernel launches across all GPUs

- **Total Score**: Sum of all GPU scores, reflecting overall computational throughput

- **Average Score per GPU**: Total Score ÷ number of GPUs

### Score Explanation

**Higher scores indicate better performance**. Scores are affected by:

1. **GPU Performance**: More powerful GPUs (more SMs, higher clock) complete kernels faster, launching more per second

2. **System Load**: External system load or GPU sharing reduces score

3. **Temperature & Power**: Thermal throttling or power limits reduce score

4. **Driver & CUDA Version**: Different versions may affect performance

### Practical Applications

- **Burn-in Testing**: Long-term runs reveal stability; score drops indicate cooling issues or hardware problems

- **Performance Comparison**: Compare same GPU models across different systems or configurations

- **Overclocking Validation**: Score should improve after overclocking; decrease suggests instability

- **Multi-GPU Balance**: Scores should be similar across GPUs; large differences indicate potential issues

### Example Score Interpretation

```
========== Performance Scores ==========
Device 0: 515 kernel launches, 51.50 launches/sec, Score: 51.50
Device 1: 522 kernel launches, 52.20 launches/sec, Score: 52.20

Total launches: 1037
Total Score: 103.70
Average Score per GPU: 51.85
========================================
```

This example shows:

- Test ran approximately 10 seconds (515 ÷ 51.50 ≈ 10)

- Device 0 completed 515 kernel launches, averaging 51.50 launches/sec

- Device 1 completed 522 kernel launches, averaging 52.20 launches/sec

- Both GPUs perform similarly (≈1.4% difference), indicating system balance

- Total score of 103.70 serves as a performance baseline for this system

## Output Example

```
[Start time] 2025-11-11 15:20:06
Detected 2 device(s), using 2 device(s). Running indefinitely. Press Ctrl+C to stop.

Device 0: NVIDIA GeForce RTX 2080 Ti | SMs=68 | streams=4 | blocks/stream=544 | tpb=256 | iters/launch=262144
Device 1: NVIDIA GeForce RTX 2080 Ti | SMs=68 | streams=4 | blocks/stream=544 | tpb=256 | iters/launch=262144
^C
Caught signal SIGINT (Ctrl+C) -> stopping gracefully... (send again to force exit)
[End time] 2025-11-11 15:20:08
Duration: 2 seconds (2 seconds)

========== Performance Scores ==========
Device 0: 112 kernel launches, 56.00 launches/sec, Score: 56.00
Device 1: 111 kernel launches, 55.50 launches/sec, Score: 55.50

Total launches: 223
Total Score: 111.50
Average Score per GPU: 55.75
========================================

Bye!
```

## LICENSE

```
MIT License

Copyright (c) 2025 Heqi Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
