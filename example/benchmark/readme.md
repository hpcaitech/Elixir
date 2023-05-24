# Benchmark

See the folder `scripts` checking how to run benchmarks.

## Environments

Here are some notice for users.

* ColossalAI and DeepSpeed can not be installed in one conda environment, just create two for them respectively.

* The version of `PyTorch` should be `1.13.1`.

* The version of `transformers` should be `4.26.1`.

* The version of `deepspeed` should be `0.8.3`.

* We found that FSDP in PyTorch is not compatible with the gradient checkpointing even in PyTorch 2.0.
Thus, we use FSDP from `fairscale(0.4.13)`.
