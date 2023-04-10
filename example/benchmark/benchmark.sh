for name_model in "opt-350m" "opt-1b" "opt-3b" "opt-7b" "opt-13b"; do
    for num_gpu in 1 2 4 8; do
        for batch_size in 8 16 24 32 40 48; do
            echo "****************** Begin ***************************"
            T_MODEL=${name_model} N_GPU=${num_gpu} N_BS=${batch_size} bash ./deepspeed.sh
            T_MODEL=${name_model} N_GPU=${num_gpu} N_BS=${batch_size} bash ./elixir.sh
            echo "****************** Finished ***************************"
            echo ""
            echo ""
        done
    done
done
