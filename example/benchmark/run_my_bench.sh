set -x

export GPUNUM=${GPUNUM:-1}
export BATCH_SIZE=${BATCH_SIZE:-28}
export MODEL_NAME=${MODEL_TYPE:-"opt-1b"}
export TRAIN_STEP=${TRAIN_STEP:-6}
# export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p benchmark_logs

torchrun --standalone --nproc_per_node=${GPUNUM} --master_port=29515 ./run_benchmark.py \
--dp_type=elixir \
--model_name=${MODEL_NAME} \
--batch_size=${BATCH_SIZE} \
--train_step=${TRAIN_STEP} \
2>&1 | tee ./benchmark_logs/${MODEL_TYPE}_gpu_${GPUNUM}_bs_${BATCH_SIZE}.log
