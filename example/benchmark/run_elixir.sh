set -x

export GPUNUM=${GPUNUM:-4}
export BATCH_SIZE=${BATCH_SIZE:-32}
export MODEL_NAME=${MODEL_TYPE:-"gpt2-400m"}
export TRAIN_STEP=${TRAIN_STEP:-6}
# export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p elixir_logs

torchrun --standalone --nproc_per_node=${GPUNUM} ./elixir_demo.py \
--model_name=${MODEL_NAME} \
--batch_size=${BATCH_SIZE} \
--train_step=${TRAIN_STEP} \
2>&1 | tee ./elixir_logs/${MODEL_TYPE}_gpu_${GPUNUM}_bs_${BATCH_SIZE}.log
