# source /opt/conda/etc/profile.d/conda.sh
conda activate adv-torch-1.13

export T_MODEL=${T_MODEL:-"gpt2-20b"}

export N_GPU=${N_GPU:-4}
export N_BS=${N_BS:-16}
export N_STEP=${N_STEP:-6}

export T_DP="elixir"
bash ./run_script.sh
