#!/bin/bash

#SBATCH -o /data/nfs_home/nchaudh1/sglang_pcl/log_sglang_gnr.%j.out
#SBATCH -D /data/nfs_home/nchaudh1/sglang_pcl
#SBATCH -J sglang
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=23:59:00

hostname

source /swtools/intel/2025.2.0/setvars.sh --force > /dev/null 2>&1
source .venv/bin/activate

export HF_HOME="/scratch/nchaudh1/huggingface"
export LD_PRELOAD=/data/swtools/intel/2025.2.0/2025.2/lib/libiomp5.so:$LD_PRELOAD
#export LD_PRELOAD=$HOME/lib/lib/libtcmalloc.so.4:$LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/nfs_home/nchaudh1/lib/lib
export LD_LIBRARY_PATH=/data/swtools/intel/2025.2.0/lib:$LD_LIBRARY_PATH
export SGLANG_USE_CPU_ENGINE=1

# export LD_PRELOAD=$HOME/jemalloc/lib/libjemalloc.so:$LD_PRELOAD
# export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

# export SGLANG_CPU_OMP_THREADS_BIND="2-41|45-84"
export SGLANG_CPU_OMP_THREADS_BIND="0-39|43-82|86-125|128-167|171-210|214-253"
# export SGLANG_CPU_OMP_THREADS_BIND="0-39|43-82|86-125|128-167"
export OMP_NUM_THREADS=40
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=50
export OMP_WAIT_POLICY=ACTIVE

export TGT_DIR=Qwen3.5-397B-A17B-GPTQ-Int4-GNR-TP6
mkdir -p $TGT_DIR
export SGLANG_TORCH_PROFILER_DIR=$HOME/sglang_pcl/profile_log

export no_proxy="localhost,127.0.0.1"

for b in 1; do
    # for dataset in gpqa_mlperf_dataset mmlupro_mlperf_dataset livecodebench_mlperf_dataset math500_mlperf_dataset; do
    for dataset in dsr1_mlperf_dataset; do
        echo "-------------------------------------"
        echo "Starting server and running benchmark with batch size: $b and dataset $dataset"
        echo "-------------------------------------"

        # if [ "$b" -le 16 ]; then
            # num_prompts=20
        # else
            # num_prompts=100
        # fi
        num_prompts=10
        
        python -m sglang.launch_server              \
            --model Qwen/Qwen3.5-397B-A17B-GPTQ-Int4                           \
            --trust-remote-code                                        \
            --disable-overlap-schedule                                 \
            --device cpu                                               \
            --host 127.0.0.1                                           \
            --port 8000                                                \
            --tp 6                                                     \
            --max-total-tokens 524288                                  \
            --enable-torch-compile                                     \
            --torch-compile-max-bs 16  &> ${TGT_DIR}/sglang_server_bs${b}_$dataset.log &

        # Wait for the server to become responsive
        echo "Waiting for vLLM server to start..."
        while ! nc -z 127.0.0.1 8000; do
        sleep 1
        done
        echo "Server is ready."

        python /scratch/savancha/bench_serving.py \
            --backend sglang \
            --model Qwen/Qwen3.5-397B-A17B-GPTQ-Int4  \
            --host 127.0.0.1 --port 8000 \
            --num-prompts $num_prompts --max-concurrency $b --output-details \
            --sharegpt-output-len 20000 --sharegpt-context-len 23140 \
            --request-rate 100 \
            --dataset-name custom_hf --dataset-path /scratch/savancha/huggingface/hub/$dataset &> ${TGT_DIR}/sglang_client_bs${b}_$dataset.log

        # Kill the background server process
        pkill -f "sglang.launch_server"
        sleep 10
        echo "Completed benchmark with batch size: $b and dataset $dataset"
    done
done

        