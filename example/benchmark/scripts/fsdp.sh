conda activate ds-torch-1.13

export T_MODEL=${T_MODEL:-"opt-1b"}

export N_GPU=${N_GPU:-1}
export N_BS=${N_BS:-16}
export N_STEP=${N_STEP:-6}

export T_DP="fsdp"
bash ./run_script.sh
