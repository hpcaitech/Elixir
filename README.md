# ElixirPlus
This repository contains an optimized implementation of Elixir which is also called Gemini in ColossalAI.

Currently, this repository is under development but you can still try some useful tools instead.

# Example

## Tools

### CUDA Memory Profiling

Function `cuda_memory_profiling` in `elixir.tracer.memory_tracer` can help you get each kind of memory occupation during training.
It tells you the CUDA memory occupation of parameters, gradient and maximum size of activations generated during training.
Moreover, it is an efficient and fast tool which enables quickly profiling OPT-175B model on a single GPU.
You can try it by yourself with the file `activation.py` in the directory `example`.
