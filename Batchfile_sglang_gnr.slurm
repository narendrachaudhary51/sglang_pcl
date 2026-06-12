#!/bin/bash

#SBATCH -o /data/nfs_home/nchaudh1/sglang_pcl/log_sglang_gnr.%j.out
#SBATCH -D /data/nfs_home/nchaudh1/sglang_pcl
#SBATCH -J sglang
#SBATCH --partition=gnrap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=5:59:00

hostname

rm -rf ~/.triton/cache/*
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

source /swtools/intel/2025.2.0/setvars.sh --force > /dev/null 2>&1
source .venv/bin/activate

export HF_HOME="/scratch/nchaudh1/huggingface"
export LD_PRELOAD=/data/swtools/intel/2025.2.0/2025.2/lib/libiomp5.so:$LD_PRELOAD
# export LD_PRELOAD=$HOME/lib/lib/libtcmalloc.so.4:/usr/lib64/libtbbmalloc.so.2:$LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/nfs_home/nchaudh1/lib/lib
export LD_LIBRARY_PATH=/data/swtools/intel/2025.2.0/lib:$LD_LIBRARY_PATH
export SGLANG_USE_CPU_ENGINE=1

export LD_PRELOAD=$HOME/jemalloc/lib/libjemalloc.so:$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:10000,muzzy_decay_ms:10000"

export SGLANG_CPU_OMP_THREADS_BIND="0-39"
# export SGLANG_CPU_OMP_THREADS_BIND="0-39|43-82|86-125|128-167|171-210|214-253"
# export SGLANG_CPU_OMP_THREADS_BIND="0-39|43-82|128-167|171-210"
export OMP_NUM_THREADS=40
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=50
export OMP_WAIT_POLICY=ACTIVE

export TGT_DIR=Qwen3-235B-A22B-W8A8-GNR-TP2
mkdir -p $TGT_DIR
export SGLANG_TORCH_PROFILER_DIR=$HOME/sglang_pcl/profile_log

export no_proxy="localhost,127.0.0.1"

for b in 1 2 4 8; do
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
        num_prompts=40

        # model="openai/gpt-oss-120b"
        model="FabioTrindade/Qwen3-235B-A22B-W8A8"

        numactl --membind="0,1" ./server_client.sh $b $num_prompts "127.0.0.1" 8000 2 "0-39|43-82" $model $dataset $TGT_DIR &
        pid1=$!
        numactl --membind="2,3" ./server_client.sh $b $num_prompts "127.0.0.1" 8001 2 "86-125|128-167" $model $dataset $TGT_DIR &
        pid2=$!
        numactl --membind="4,5" ./server_client.sh $b $num_prompts "127.0.0.1" 8002 2 "171-210|214-253" $model $dataset $TGT_DIR &
        pid3=$!
        # ./server_client.sh $b $num_prompts 8003 1 "128-167" $model $dataset $TGT_DIR &
        # ./server_client.sh $b $num_prompts 8004 1 "171-210" $model $dataset $TGT_DIR &
        # ./server_client.sh $b $num_prompts 8005 1 "214-253" $model $dataset $TGT_DIR &

        wait $pid1 
        wait $pid2 
        wait $pid3
        echo "Completed benchmark with batch size: $b and dataset $dataset on all cores"
        
        # python -m sglang.launch_server              \
        #     --model openai/gpt-oss-120b                           \
        #     --trust-remote-code                                        \
        #     --disable-overlap-schedule                                 \
        #     --device cpu                                               \
        #     --host 127.0.0.1                                           \
        #     --port 8000                                                \
        #     --tp 1                                                     \
        #     --max-total-tokens 524288                               \
        #     --enable-torch-compile                                     \
        #     --torch-compile-max-bs 16  &> ${TGT_DIR}/sglang_server_bs${b}_$dataset.log &
        # SERVER_PID=$!
        # # Wait for the server to become responsive
        # echo "Waiting for vLLM server to start..."
        # # while ! nc -z 127.0.0.1 8000; do
        # # sleep 1
        # # done
        # while ! curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do sleep 2; done
        # echo "Server is ready."

        # # python3 -m sglang.test.send_one --profile --profile-steps 5 --profile-by-stage \
        # #     --host 127.0.0.1 --port 8000 \
        # #     &> ${TGT_DIR}/sglang_profile_bs${b}_${dataset}.log

        # python /scratch/savancha/bench_serving.py \
        #     --backend sglang \
        #     --model openai/gpt-oss-120b  \
        #     --host 127.0.0.1 --port 8000 \
        #     --num-prompts $num_prompts --max-concurrency $b --output-details \
        #     --sharegpt-output-len 20000 --sharegpt-context-len 23140 \
        #     --request-rate 100 \
        #     --dataset-name custom_hf --dataset-path /scratch/savancha/huggingface/hub/$dataset &> ${TGT_DIR}/sglang_client_bs${b}_$dataset.log

        # # Kill the background server process
        # pkill -f "sglang.launch_server"
        # pkill -TERM -P "$SERVER_PID" 2>/dev/null
        # kill -TERM "$SERVER_PID" 2>/dev/null
        # sleep 10
        # # wait until port 8000 is really free
        # while ss -ltn 'sport = :8000' | grep -q LISTEN; do sleep 1; done
        # sleep 5
        # echo "Completed benchmark with batch size: $b and dataset $dataset"
    done
done

        