# Busy GPU

## 编译

```bash
nvcc -O3 -use_fast_math busy_gpu.cu -o busy_gpu

# 指定SM
nvcc -O3 -use_fast_math busy_gpu.cu -o busy_gpu \
  -cudart static \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89
```

## 运行

```bash
./busy_gpu
timeout -s SIGINT 12h ./busy_gpu  # 运行12小时
```