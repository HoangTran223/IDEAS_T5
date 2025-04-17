#! /bin/bash
set -e
# export CUDA_VISIBLE_DEVICES=${0}
MASTER_ADDR=localhost
MASTER_PORT=${1}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${2}

if ! command -v nvidia-smi &> /dev/null || [ $(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l) -eq 0 ] || [ $(nvidia-smi --query-gpu=count --format=csv,noheader) -eq 0 ]; then
    echo "Error: No GPU detected. Please check NVIDIA driver and CUDA installation."
    echo "Falling back to CPU (not recommended for evaluation)."
    DISTRIBUTED_ARGS=""
    USE_DEEPSPEED=false
    DEVICE="cpu"
else
    echo "GPU detected, using GPU 0"
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                      --nnodes $NNODES \
                      --node_rank $NODE_RANK \
                      --master_addr $MASTER_ADDR \
                      --master_port $MASTER_PORT"
    USE_DEEPSPEED=true
    DEVICE="cuda:0"
    export CUDA_VISIBLE_DEVICES=0  # Chỉ định GPU 0
fi

# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
#                   --nnodes $NNODES \
#                   --node_rank $NODE_RANK \
#                   --master_addr $MASTER_ADDR \
#                   --master_port $MASTER_PORT"

# model
BASE_PATH=/home/mcn/tue_x/DSKD
CKPT_PATH=${3}
# CKPT_SETTING=$(echo ${CKPT_PATH} | awk -F'/' '{print $(NF-4)"/"$(NF-3)"/"$(NF-2)"/"$(NF-1)}')
# MODEL_TYPE=$(echo ${CKPT_PATH} | awk -F'/' '{print $(NF-4)}')
MODEL_TYPE="gpt2"
# task
TASK="eval_main"
# data
DATA_NAME=${4}
DATA_DIR="${BASE_PATH}/data/${DATA_NAME}"
# DATA_DIR="${BASE_PATH}/processed_data/${DATA_NAME}/full/gpt2"
DATA_NUM=${7--1}
# hp
EVAL_BATCH_SIZE=${5}
SEED=${6}
# runtime
SAVE_PATH=$(dirname ${CKPT_PATH})

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type ${MODEL_TYPE}"
# task
OPTS+=" --task ${TASK}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAME}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num ${DATA_NUM}"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/evaluate.py ${OPTS}"
echo ${CMD}

${CMD}
