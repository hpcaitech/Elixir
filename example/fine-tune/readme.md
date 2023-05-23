# Fine-tune Example

## Command

```bash
bash run_elixir.sh (or run_ddp.sh)
```

## Results


| config | accuracy | f1 |
| :----: | :----:   | :----: |
| ddp-fp32-1GPUs    | 0.8382 | 0.8889 |
| elixir-fp16-1GPUs | 0.8456 | 0.8923 |
| ddp-fp32-2GPUs    | 0.8333 | 0.8859 |
| elixir-fp16-2GPUs | 0.8725 | 0.9107 |
| ddp-fp32-4GPUs    | 0.8358 | 0.8874 |
| elixir-fp16-4GPUs | 0.8480 | 0.8920 |
