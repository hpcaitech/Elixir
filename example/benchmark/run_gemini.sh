set -x

# The following options only valid when DISTPLAN="colossalai"
export GPUNUM=${GPUNUM:-4}
export PLACE_POLICY=${PLACEMENT:-"cuda"}
export BATCH_SIZE=${BATCH_SIZE:-32}
export MODEL_NAME=${MODEL_TYPE:-"gpt2-400m"}
export TRAIN_STEP=${TRAIN_STEP:-6}
# export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p gemini_logs

torchrun --standalone --nproc_per_node=${GPUNUM} ./gemini_demo.py \
--model_name=${MODEL_NAME} \
--batch_size=${BATCH_SIZE} \
--place_policy=${PLACE_POLICY} \
--train_step=${TRAIN_STEP} \
2>&1 | tee ./gemini_logs/${MODEL_TYPE}_gpu_${GPUNUM}_bs_${BATCH_SIZE}_${PLACEMENT}.log
