#!/bin/bash

#SBATCH -o /data/nfs_home/nchaudh1/sglang_pcl/log_sglang_emr.%j.out
#SBATCH -D /data/nfs_home/nchaudh1/sglang_pcl
#SBATCH -J sglang
#SBATCH --partition=emr
#SBATCH --constraint="ddr5600"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=23:59:00

hostname

source /swtools/intel/2025.2.0/setvars.sh --force > /dev/null 2>&1
source .venv/bin/activate

export HF_HOME="/scratch/nchaudh1/huggingface"

export LD_PRELOAD=/data/swtools/intel/2025.2.0/2025.2/lib/libiomp5.so:$LD_PRELOAD
#export LD_PRELOAD=$HOME/lib/lib/libtcmalloc.so.4:/usr/lib64/libtbbmalloc.so.2:$LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/nfs_home/nchaudh1/lib/lib
export LD_LIBRARY_PATH=/data/swtools/intel/2025.2.0/lib:$LD_LIBRARY_PATH
export SGLANG_USE_CPU_ENGINE=1

export LIBXSMM_ROOT=/data/nfs_home/nchaudh1/libxsmm
export LD_LIBRARY_PATH=$LIBXSMM_ROOT/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=$HOME/jemalloc/lib/libjemalloc.so:$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"

export no_proxy="localhost,127.0.0.1"

export SGLANG_CPU_OMP_THREADS_BIND="0-63|64-127"
export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=50
export OMP_WAIT_POLICY=ACTIVE

export TGT_DIR=gpt-oss-120b-EMR-TP1-server2-evangelos
mkdir -p $TGT_DIR
export SGLANG_CPU_MXFP4_LIBXSMM=1

for b in 1 2 4 8; do
    # for dataset in gpqa_mlperf_dataset mmlupro_mlperf_dataset livecodebench_mlperf_dataset math500_mlperf_dataset; do
    for dataset in dsr1_mlperf_dataset; do
        echo "-------------------------------------"
        echo "Starting server and running benchmark with batch size: $b and dataset $dataset"
        echo "-------------------------------------"

        # if [ "$b" -le 16 ]; then
        #     num_prompts=20
        # else
        #     num_prompts=100
        # fi
        num_prompts=20
        model="openai/gpt-oss-120b"

        numactl --membind="0" ./server_client.sh $b $num_prompts "127.0.0.1" 8000 1 "0-63" $model $dataset $TGT_DIR &
        pid1=$!
        # numactl --membind="1" ./server_client.sh $b $num_prompts "127.0.0.1" 8001 1 "64-127" $model $dataset $TGT_DIR &
        # pid2=$!

        wait $pid1
        echo "Completed benchmark with batch size: $b and dataset $dataset on all cores"

        
        # python -u -m sglang.launch_server              \
        #     --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
        #     --trust-remote-code                                        \
        #     --disable-overlap-schedule                                 \
        #     --device cpu                                               \
        #     --host 127.0.0.1                                           \
        #     --port 8000                                                \
        #     --tp 2                                                     \
        #     --max-total-tokens 32768                                   
        #     # --enable-torch-compile                                     \
        #     # --torch-compile-max-bs 16  &> ${TGT_DIR}/sglang_server_bs${b}_$dataset.log &

        # # Wait for the server to become responsive
        # echo "Waiting for vLLM server to start..."
        # while ! nc -z 127.0.0.1 8000; do
        # sleep 1
        # done
        # echo "Server is ready."

        # python /scratch/savancha/bench_serving.py \
        #     --backend sglang \
        #     --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
        #     --host 127.0.0.1 --port 8000 \
        #     --num-prompts $num_prompts --max-concurrency $b --output-details \
        #     --sharegpt-output-len 20000 --sharegpt-context-len 23140 \
        #     --request-rate 100 \
        #     --dataset-name custom_hf --dataset-path /scratch/savancha/huggingface/hub/$dataset &> ${TGT_DIR}/sglang_client_bs${b}_$dataset.log

        # # Kill the background server process
        # pkill -f "sglang.launch_server"
        # sleep 10
        # echo "Completed benchmark with batch size: $b and dataset $dataset"
    done
done



# TRANSFORMERS_OFFLINE=1 python -m sglang.bench_serving   \
#     --dataset-name random        \
#     --random-input-len 2048      \
#     --random-output-len 2048     \
#     --num-prompts 1              \
#     --request-rate inf           \
#     --random-range-ratio 1.0
 
# TRANSFORMERS_OFFLINE=1 python -m sglang.bench_serving \
#         --backend sglang \
#         --model deepseek-ai/DeepSeek-R1-0528 \
#         --host 0.0.0.0 --port 30000 \
#         --num-prompts 500 --max-concurrency 1 --output-details \
#         --sharegpt-output-len 2000 --sharegpt-context-len 23140 \
#         --request-rate 100 \
#         --dataset-name sharegpt
        # --dataset-name custom_hf --dataset-path $HOME/dsr1_mlperf_dataset \
        
        