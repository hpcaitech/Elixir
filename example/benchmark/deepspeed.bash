source /opt/lcsoftware/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.3.0/miniconda3-4.10.3-u6p3tgreee7aigtnvuhr44yqo7vcg6r6/etc/profile.d/conda.sh
conda activate ds-torch-1.13

export T_MODEL=${T_MODEL:-"opt-1b"}

export N_GPU=${N_GPU:-1}
export N_BS=${N_BS:-16}
export N_STEP=${TRAIN_STEP:-6}

export T_DP="zero2"
bash ./run_script.sh

export T_DP="zero2-offload"
bash ./run_script.sh

export T_DP="zero3"
bash ./run_script.sh

export T_DP="zero3-offload"
bash ./run_script.sh
