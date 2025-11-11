# Busy GPU

[English document](./README.md) | 中文文档

一个 GPU 压力测试和烤机工具，用于测试 GPU 的稳定性和性能。

## 编译

使用 Makefile（推荐）：

```bash
make          # 编译
make clean    # 清理
make rebuild  # 重新编译
make install  # 安装到系统（需要 sudo）
```

或者手动编译：

```bash
nvcc -O3 -use_fast_math busy_gpu.cu -o busy_gpu

# 指定SM
nvcc -O3 -use_fast_math busy_gpu.cu -o busy_gpu \
  -cudart static \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89
```

## 使用方法

### 基本用法

```bash
# 无限运行，使用所有 GPU
./busy_gpu

# 查看帮助
./busy_gpu --help

# 查看版本信息
./busy_gpu --version

# 指定运行时间
./busy_gpu -t 30s    # 运行30秒
./busy_gpu -t 5m     # 运行5分钟
./busy_gpu -t 2h     # 运行2小时
./busy_gpu -t 1d     # 运行1天

# 指定使用的 GPU
./busy_gpu -d 0,2    # 只使用 GPU 0 和 2

# 组合使用
./busy_gpu -t 1h -d 0,1    # 在 GPU 0 和 1 上运行1小时
```

### 环境变量

可以通过环境变量调整程序行为：

```bash
# 设置每个设备的 CUDA 流数量（默认：4）
BURN_STREAMS=8 ./busy_gpu

# 设置每个块的线程数（默认：256）
BURN_TPB=512 ./busy_gpu

# 设置每次 kernel 启动的迭代次数（默认：262144）
BURN_ITERS=524288 ./busy_gpu

# 限制可见的 GPU
CUDA_VISIBLE_DEVICES=0,1 ./busy_gpu
```

### 使用 timeout 命令

```bash
# 使用 timeout 命令运行12小时
timeout 12h ./busy_gpu
```

## 功能特性

- ✅ 多 GPU 并行压力测试
- ✅ 性能评分系统
- ✅ 支持定时运行
- ✅ 灵活的设备选择
- ✅ 优雅的信号处理
- ✅ 详细的运行统计

## 工作原理

### 压力测试机制

Busy GPU 通过持续启动 CUDA kernel 来对 GPU 施加计算压力，其核心工作原理如下：

1. **多流并发执行**
   - 每个 GPU 创建多个 CUDA 流（默认 4 个）
   - 多个流可以并发执行，充分利用 GPU 的并行能力
   - 使用事件（Event）机制来检测 kernel 完成状态

2. **计算密集型 Kernel**
   ```cuda
   // 每个线程执行大量浮点运算
   for (int i = 0; i < iters_per_launch; ++i) {
       x = x * 1.0000001f + 1.0f;    // 浮点乘加运算
       x = x * 0.9999999f + 0.5f;
       x = x * 1.0000002f + 0.25f;
       x = x * 0.9999998f + 0.125f;
   }
   ```
   - 每次 kernel 启动执行 262,144 次迭代（可配置）
   - 每次迭代包含 4 组浮点乘加运算
   - 总计每个线程执行超过 100 万次浮点运算

3. **资源占用策略**
   - 根据 GPU 的 SM（Streaming Multiprocessor）数量动态调整 block 数量
   - 默认每个 SM 分配 32 个 block，确保充分利用硬件资源
   - 每个 block 包含 256 个线程（可配置）

4. **连续工作循环**
   ```
   初始化 → 预热启动 → 循环检测
                          ↓
            ← 启动新 kernel ← 事件完成？
   ```
   - 使用非阻塞方式查询 kernel 完成状态
   - 一旦某个流的 kernel 完成，立即启动新的 kernel
   - 保持 GPU 始终处于满载工作状态

### 为什么能有效测试 GPU？

1. **全面的硬件覆盖**
   - 同时使用所有 SM 单元
   - 大量浮点运算覆盖 ALU（算术逻辑单元）
   - 频繁的内存访问测试显存和内存控制器

2. **持续的高功耗**
   - 密集的计算保持 GPU 核心高频运行
   - 产生大量热量，测试散热系统
   - 长时间运行可发现温度相关的稳定性问题

3. **可重现的负载**
   - 相同配置下的运算量是确定的
   - 性能评分可用于对比不同系统或配置
   - 评分下降可能表示硬件问题或过热降频

## 信号处理

程序支持以下信号：

- `Ctrl+C` (SIGINT)：优雅停止并显示统计
- `SIGTERM`：优雅停止
- `SIGHUP`：优雅停止
- `SIGQUIT`：优雅停止

发送信号两次可强制立即退出。

## 性能指标说明

程序结束时会输出详细的性能统计信息，包括以下指标：

### 单卡指标
- **kernel launches**：该 GPU 在运行期间启动的 kernel 总次数
- **launches/sec**：每秒平均启动的 kernel 次数（kernel launches ÷ 运行时长）
- **Score**：单卡性能评分，等于 launches/sec

### 整体指标
- **Total launches**：所有 GPU 的 kernel 启动次数总和
- **Total Score**：所有 GPU 的评分总和，反映整体计算吞吐能力
- **Average Score per GPU**：平均每个 GPU 的评分（Total Score ÷ GPU 数量）

### 评分说明

**评分越高表示性能越好**。评分受以下因素影响：

1. **GPU 性能**：更强大的 GPU（更多 SM、更高频率）可以更快完成 kernel 执行，从而在单位时间内启动更多 kernel
2. **系统负载**：如果系统有其他负载或 GPU 被其他程序占用，评分会降低
3. **温度和功耗**：GPU 过热或达到功耗限制时会降频，导致评分下降
4. **驱动和 CUDA 版本**：不同版本的驱动和 CUDA 可能有性能差异

### 实际应用

- **烤机测试**：长时间运行观察评分是否稳定，评分下降可能表示散热不良或硬件问题
- **性能对比**：相同 GPU 型号在不同系统或配置下的评分对比
- **超频验证**：超频后评分应该提升，如果评分反而下降可能是不稳定的表现
- **多卡均衡**：在多 GPU 系统中，各卡评分应该接近，差异过大可能表示某些卡有问题

### 示例解读

```
========== Performance Scores ==========
Device 0: 515 kernel launches, 51.50 launches/sec, Score: 51.50
Device 1: 522 kernel launches, 52.20 launches/sec, Score: 52.20

Total launches: 1037
Total Score: 103.70
Average Score per GPU: 51.85
========================================
```

这个例子表示：
- 测试运行了约 10 秒（515 ÷ 51.50 ≈ 10）
- Device 0 完成了 515 次 kernel 启动，平均每秒 51.50 次
- Device 1 完成了 522 次 kernel 启动，平均每秒 52.20 次
- 两块 GPU 性能基本一致（差异约 1.4%），说明系统均衡
- 总体评分 103.70，可以作为这套系统的性能基准



## 输出示例

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

## 许可证

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
