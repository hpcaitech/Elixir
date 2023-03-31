export GPU_NUM=${GPU_NUM:-8}
wc=`cat /proc/cpuinfo | grep "processor"| wc -l`
let TNUM=wc/${GPU_NUM}
env OMP_NUM_THREADS=${TNUM} torchrun --nproc_per_node=${GPU_NUM} --master_port=29515 profile_optimizer.py
