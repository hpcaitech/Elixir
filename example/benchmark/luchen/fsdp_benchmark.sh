export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for name_model in "gpt2-4b" "gpt2-10b" "gpt2-15b" "gpt2-20b"; do
    for num_gpu in 1 2 4; do
        for batch_size in 4 8 12 16; do
            echo "****************** Begin ***************************"
            T_MODEL=${name_model} N_GPU=${num_gpu} N_BS=${batch_size} bash ./fsdp.sh
            echo "****************** Finished ***************************"
            echo ""
            echo ""
        done
    done
done
