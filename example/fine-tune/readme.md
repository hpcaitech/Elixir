# Fine-tune Example

## Command

```bash
bash run_elixir.sh (or run_ddp.sh)
```

## Results


| config | accuracy | f1 |
| :----: | :----:   | :----: |
| ddp-fp32-1GPUs    | 0.8382 | 0.8889 |
| elixir-fp32-1GPUs | 0.8407 | 0.8904 |
| ddp-fp32-2GPUs    | 0.8333 | 0.8859 |
| elixir-fp32-2GPUs | 0.8333 | 0.8855 |
| ddp-fp32-4GPUs    | 0.8358 | 0.8874 |
| elixir-fp32-4GPUs | 0.8382 | 0.8889 |
