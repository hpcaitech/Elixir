export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for name_model in "opt-1b" "opt-3b" "opt-7b" "opt-13b"; do
    for num_gpu in 1 2 4; do
        for batch_size in 8 16 24 32; do
            echo "****************** Begin ***************************"
            T_MODEL=${name_model} N_GPU=${num_gpu} N_BS=${batch_size} bash ./deepspeed.sh
            T_MODEL=${name_model} N_GPU=${num_gpu} N_BS=${batch_size} bash ./elixir.sh
            echo "****************** Finished ***************************"
            echo ""
            echo ""
        done
    done
done
