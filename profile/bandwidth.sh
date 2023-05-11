for num_gpu in 1 2 4; do
    echo "${num_gpu} is used"
    torchrun --nproc_per_node=${num_gpu} --master_port=29515 profile_bandwidth.py
done
