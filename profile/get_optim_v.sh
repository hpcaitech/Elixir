for num_gpu in 1 2 4; do
    echo "${num_gpu} is used"
    wc=`cat /proc/cpuinfo | grep "processor"| wc -l`
    let TNUM=wc/${num_gpu}
    env OMP_NUM_THREADS=${TNUM} torchrun --nproc_per_node=${num_gpu} --master_port=29515 profile_optimizer.py
done
