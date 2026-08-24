#!/bin/bash

b=${1:-1}
num_prompts=${2:-20}
host=${3:-"127.0.0.1"}
port=${4:-8000}
TP=${5:-1}
core_list=${6:-"0-39"}
model=${7:-"openai/gpt-oss-120b"}
dataset=${8:-"dsr1_mlperf_dataset"}
TGT_DIR=${9:-"gpt-oss-120b-GNR-TP1"}

# create a directory for torch inductor and triton compiler
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache/${model}/${core_list}
export TRITON_CACHE_DIR=/tmp/triton_cache/${model}/${core_list}
mkdir -p $TORCHINDUCTOR_CACHE_DIR
mkdir -p $TRITON_CACHE_DIR

export MASTER_ADDR="${host}"
export MASTER_PORT="${port}"

export TIKTOKEN_ENCODINGS_BASE=/scratch/nchaudh1/tiktoken_encodings

PYTHONOPTIMIZE=1 SGLANG_CPU_OMP_THREADS_BIND=${core_list} python -m sglang.launch_server              \
        --model ${model}                                           \
        --quantization w8a8_int8                                       \
        --dtype bfloat16                                          \
        --trust-remote-code                                        \
        --disable-overlap-schedule                                 \
        --device cpu                                               \
        --host ${host}                                             \
        --port ${port}                                             \
        --tp ${TP}                                                 \
        --max-total-tokens 131072                                  \
        --enable-torch-compile                                     \
        --torch-compile-max-bs 1  &> ${TGT_DIR}/sglang_server_bs${b}_${dataset}_${core_list}.log &

SERVER_PID=$!
# Wait for the server to become responsive
echo "Waiting for sglang server to start..."
while ! curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; do sleep 2; done
echo "Server is ready."

# sudo /etc/pcl_cleanup_memory.sh

SGLANG_CPU_OMP_THREADS_BIND=${core_list} python bench_serving.py     \
    --backend sglang                                                 \
    --model ${model}                                                 \
    --host ${host} --port ${port}                                    \
    --num-prompts $num_prompts --max-concurrency $b --output-details \
    --sharegpt-output-len 20000 --sharegpt-context-len 23140         \
    --request-rate 100                                               \
    --seed ${port}                                                   \
    --disable-ignore-eos                                             \
    --dataset-name custom_hf --dataset-path /cold_storage/ml_datasets/narendra/huggingface/hub/$dataset &> ${TGT_DIR}/sglang_client_bs${b}_${dataset}_${core_list}.log

rm -rf $TORCHINDUCTOR_CACHE_DIR
rm -rf $TRITON_CACHE_DIR

# replace `wait $SERVER_PID` with:
kill -TERM "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
# wait for the port to be released before the next iteration reuses it
while ss -ltn "sport = :${port}" | grep -q LISTEN; do sleep 1; done

echo "Completed benchmark with batch size: $b and dataset $dataset on cores ${core_list}"