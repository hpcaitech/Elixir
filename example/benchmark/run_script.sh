set -x

T_DP=${T_DP:-"fsdp"}
T_MODEL=${T_MODEL:-"opt-1b"}

N_GPU=${N_GPU:-1}
N_BS=${N_BS:-16}
N_STEP=${N_STEP:-6}

mkdir -p benchmark_logs

wc=`cat /proc/cpuinfo | grep "processor"| wc -l`
let TNUM=wc/${N_GPU}

env OMP_NUM_THREADS=${TNUM} torchrun --nproc_per_node=${N_GPU} --master_port=29911 ./script.py \
--dp_type=${T_DP} \
--model_name=${T_MODEL} \
--batch_size=${N_BS} \
--train_step=${N_STEP} \
2>&1 | tee ./benchmark_logs/${T_MODEL}_bs_${N_BS}_gpu_${N_GPU}_${T_DP}.log
