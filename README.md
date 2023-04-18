# Elixir (Gemini2.0)
Elixir, also known as Gemini, is a technology designed to facilitate the training of large models on a small GPU cluster.
Its goal is to eliminate data redundancy and leverage CPU memory to accommodate really large models.
In addition, Elixir automatically profiles each training step prior to execution and selects the optimal configuration for the ratio of redundancy and the device for each parameter.
While this repository is currently under development, users can still experiment with some useful tools and beta APIs.

# Environment

This version is a beta release, so the running environment is somewhat restrictive.
We are only demonstrating our running environment here, as we have not yet tested its compatibility.
We have set the CUDA version to `11.6` and the PyTorch version to `1.13.1+cu11.6`.

Three dependent package should be installed from source.
- [ColossalAI](https://github.com/hpcaitech/ColossalAI) (necessary): just clone it and use `pip install .` from the newest master branch.
- [Apex](https://github.com/NVIDIA/apex) (optional): clone it, checkout to tag `22.03`, and install it.
- [Xformers](https://github.com/facebookresearch/xformers) (optional): clone it, checkout to tag `v0.0.17`, and install it.

Finally, install all packages in the `requirements.txt`.

# Example

## Tools

### CUDA Memory Profiling

Function `cuda_memory_profiling` in `elixir.tracer.memory_tracer` can help you profile each kind of memory occupation during training.
It tells you the CUDA memory occupation of parameters, gradient and maximum size of activations generated during training.
Moreover, it is an efficient and fast tool which enables quickly profiling OPT-175B model on a single GPU.
You can try it by yourself with the file `activation.py` in the directory `example`.

(I think you should have at least 16GB CUDA memory to run the OPT-175B example but that doesn't matter. Just try it first.)


## APIs

Check out how to wrap your model and optimizer [here](https://github.com/hpcaitech/Elixir/blob/main/example/common/elx.py).
