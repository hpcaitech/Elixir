source /opt/lcsoftware/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.3.0/miniconda3-4.10.3-u6p3tgreee7aigtnvuhr44yqo7vcg6r6/etc/profile.d/conda.sh
conda activate fsdp-torch-2.0

export T_MODEL=${T_MODEL:-"opt-1b"}

export N_GPU=${N_GPU:-1}
export N_BS=${N_BS:-16}
export N_STEP=${N_STEP:-6}

export T_DP="fsdp"
bash ./run_script.sh
