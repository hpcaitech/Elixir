module load miniconda3
conda activate elx-ds

export T_MODEL=${T_MODEL:-"opt-350m"}

export N_GPU=${N_GPU:-1}
export N_BS=${N_BS:-16}
export N_STEP=${N_STEP:-6}

export T_DP="zero2"
bash ./rm_lock.sh
bash ./run_script.sh

export T_DP="zero2-offload"
bash ./rm_lock.sh
bash ./run_script.sh

export T_DP="zero3"
bash ./rm_lock.sh
bash ./run_script.sh

export T_DP="zero3-offload"
bash ./rm_lock.sh
bash ./run_script.sh
