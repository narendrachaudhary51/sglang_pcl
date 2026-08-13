#!/bin/bash

# take script name as argument
bench_script=$1

# exit if there is no argument
if [ -z "$bench_script" ]; then
    echo "Usage: $0 <benchmark_script.py>"
    exit 1
fi

source /swtools/intel/2025.2.0/setvars.sh --force > /dev/null 2>&1

source $HOME/sglang_pcl/.venv/bin/activate
export KMP_AFFINITY=granularity=fine,compact,1,0

export LD_PRELOAD=/data/swtools/intel/2025.2.0/2025.2/lib/libiomp5.so:$LD_PRELOAD
# export LD_PRELOAD=$HOME/lib/lib/libtcmalloc.so.4:/usr/lib64/libtbbmalloc.so.2:$LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/nfs_home/nchaudh1/lib/lib
export LD_LIBRARY_PATH=/data/swtools/intel/2025.2.0/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=$HOME/jemalloc/lib/libjemalloc.so:$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
# export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:10000,muzzy_decay_ms:10000"

# get cores info on socket 0 and use it for thread numbers
cores=$(lscpu -p=CPU,SOCKET | grep -v '^#' | grep ',0' | wc -l)
cores=$((cores / 2)) # use half of the cores on socket 0 for the benchmark

export OMP_NUM_THREADS=$cores
export SGLANG_USE_CPU_ENGINE=1
echo "Number of real cores on socket 0: $cores"
echo "Setting OMP_NUM_THREADS to $OMP_NUM_THREADS"

numactl -m 0 -C 0-$((cores - 1)) python $bench_script "${@:2}"