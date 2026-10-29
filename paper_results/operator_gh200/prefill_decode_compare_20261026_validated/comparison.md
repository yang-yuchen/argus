# GH200 causal prefill and decode comparison

Baselines use argus-gh200-latest; Nautilus uses the prior Neptune TVM/Triton runtime and measurements from different GH200 nodes. Ratios are cross-run comparisons. Times are mean sums of GPU kernel durations per cold-L2 NVTX repeat; compilation, setup and CPU launch time are excluded.

Local Neptune TileLang is the repository runner, distinct from the exported Nautilus kernels. Its decode runner inherits the same TileLang implementation.

## Fastest measured baseline per length

| Case | Baseline | Baseline us | Nautilus us | Baseline/Nautilus |
|---|---|---:|---:|---:|
| decode_L256 | TileLang | 6.678 | 5.341 | 1.250x |
| decode_L512 | Local Neptune TileLang | 7.647 | 6.706 | 1.140x |
| decode_L1024 | TileLang | 11.194 | 10.272 | 1.090x |
| decode_L2048 | TileLang | 17.811 | 17.665 | 1.008x |
| decode_L4096 | TileLang | 30.362 | 30.188 | 1.006x |
| decode_L8192 | TileLang | 50.119 | 50.397 | 0.994x |
| decode_L16384 | TileLang | 85.290 | 89.344 | 0.955x |
| decode_L32768 | TileLang | 154.874 | 163.552 | 0.947x |
| prefillc_L256 | TileLang | 8.580 | 7.812 | 1.098x |
| prefillc_L512 | cuDNN SDPA | 13.080 | 15.020 | 0.871x |
| prefillc_L1024 | cuDNN SDPA | 30.810 | 34.346 | 0.897x |
| prefillc_L2048 | cuDNN SDPA | 84.616 | 106.853 | 0.792x |
| prefillc_L4096 | cuDNN SDPA | 290.988 | 344.299 | 0.845x |
| prefillc_L8192 | cuDNN SDPA | 984.541 | 1154.235 | 0.853x |
| prefillc_L16384 | cuDNN SDPA | 3668.873 | 4253.793 | 0.862x |
| prefillc_L32768 | cuDNN SDPA | 14260.957 | 16312.898 | 0.874x |

## All measured baselines

| Case | Baseline | Baseline us | Nautilus us | Baseline/Nautilus |
|---|---|---:|---:|---:|
| decode_L256 | FlashInfer | 7.560 | 5.341 | 1.416x |
| decode_L256 | FlexAttention | 10.791 | 5.341 | 2.020x |
| decode_L256 | Local Neptune TileLang | 6.682 | 5.341 | 1.251x |
| decode_L256 | TileLang | 6.678 | 5.341 | 1.250x |
| decode_L256 | cuDNN SDPA | 7.607 | 5.341 | 1.424x |
| decode_L256 | PyTorch Flash | 9.428 | 5.341 | 1.765x |
| decode_L256 | FlashAttention-2 | 9.013 | 5.341 | 1.688x |
| decode_L512 | FlashInfer | 11.164 | 6.706 | 1.665x |
| decode_L512 | FlexAttention | 14.225 | 6.706 | 2.121x |
| decode_L512 | Local Neptune TileLang | 7.647 | 6.706 | 1.140x |
| decode_L512 | TileLang | 7.745 | 6.706 | 1.155x |
| decode_L512 | cuDNN SDPA | 9.217 | 6.706 | 1.374x |
| decode_L512 | PyTorch Flash | 11.079 | 6.706 | 1.652x |
| decode_L512 | FlashAttention-2 | 10.496 | 6.706 | 1.565x |
| decode_L1024 | FlashInfer | 18.613 | 10.272 | 1.812x |
| decode_L1024 | FlexAttention | 21.474 | 10.272 | 2.091x |
| decode_L1024 | Local Neptune TileLang | 11.395 | 10.272 | 1.109x |
| decode_L1024 | TileLang | 11.194 | 10.272 | 1.090x |
| decode_L1024 | cuDNN SDPA | 12.426 | 10.272 | 1.210x |
| decode_L1024 | PyTorch Flash | 15.655 | 10.272 | 1.524x |
| decode_L1024 | FlashAttention-2 | 15.350 | 10.272 | 1.494x |
| decode_L2048 | FlashInfer | 34.180 | 17.665 | 1.935x |
| decode_L2048 | FlexAttention | 30.640 | 17.665 | 1.734x |
| decode_L2048 | Local Neptune TileLang | 17.882 | 17.665 | 1.012x |
| decode_L2048 | TileLang | 17.811 | 17.665 | 1.008x |
| decode_L2048 | cuDNN SDPA | 19.071 | 17.665 | 1.080x |
| decode_L2048 | PyTorch Flash | 22.152 | 17.665 | 1.254x |
| decode_L2048 | FlashAttention-2 | 21.729 | 17.665 | 1.230x |
| decode_L4096 | FlashInfer | 63.798 | 30.188 | 2.113x |
| decode_L4096 | FlexAttention | 53.442 | 30.188 | 1.770x |
| decode_L4096 | Local Neptune TileLang | 30.579 | 30.188 | 1.013x |
| decode_L4096 | TileLang | 30.362 | 30.188 | 1.006x |
| decode_L4096 | cuDNN SDPA | 30.806 | 30.188 | 1.021x |
| decode_L4096 | PyTorch Flash | 34.349 | 30.188 | 1.138x |
| decode_L4096 | FlashAttention-2 | 33.946 | 30.188 | 1.124x |
| decode_L8192 | FlashInfer | 113.948 | 50.397 | 2.261x |
| decode_L8192 | FlexAttention | 99.640 | 50.397 | 1.977x |
| decode_L8192 | Local Neptune TileLang | 50.264 | 50.397 | 0.997x |
| decode_L8192 | TileLang | 50.119 | 50.397 | 0.994x |
| decode_L8192 | cuDNN SDPA | 51.119 | 50.397 | 1.014x |
| decode_L8192 | PyTorch Flash | 55.796 | 50.397 | 1.107x |
| decode_L8192 | FlashAttention-2 | 55.508 | 50.397 | 1.101x |
| decode_L16384 | FlashInfer | 213.229 | 89.344 | 2.387x |
| decode_L16384 | FlexAttention | 191.023 | 89.344 | 2.138x |
| decode_L16384 | Local Neptune TileLang | 85.308 | 89.344 | 0.955x |
| decode_L16384 | TileLang | 85.290 | 89.344 | 0.955x |
| decode_L16384 | cuDNN SDPA | 87.723 | 89.344 | 0.982x |
| decode_L16384 | PyTorch Flash | 95.364 | 89.344 | 1.067x |
| decode_L16384 | FlashAttention-2 | 94.079 | 89.344 | 1.053x |
| decode_L32768 | FlashInfer | 410.134 | 163.552 | 2.508x |
| decode_L32768 | FlexAttention | 372.031 | 163.552 | 2.275x |
| decode_L32768 | Local Neptune TileLang | 154.911 | 163.552 | 0.947x |
| decode_L32768 | TileLang | 154.874 | 163.552 | 0.947x |
| decode_L32768 | cuDNN SDPA | 157.159 | 163.552 | 0.961x |
| decode_L32768 | PyTorch Flash | 169.065 | 163.552 | 1.034x |
| decode_L32768 | FlashAttention-2 | 166.257 | 163.552 | 1.017x |
| prefillc_L256 | FlexAttention | 9.560 | 7.812 | 1.224x |
| prefillc_L256 | Local Neptune TileLang | 8.818 | 7.812 | 1.129x |
| prefillc_L256 | TAWA | 12.463 | 7.812 | 1.595x |
| prefillc_L256 | TileLang | 8.580 | 7.812 | 1.098x |
| prefillc_L256 | cuDNN SDPA | 8.977 | 7.812 | 1.149x |
| prefillc_L256 | PyTorch Flash | 18.662 | 7.812 | 2.389x |
| prefillc_L256 | FlashAttention-2 | 18.110 | 7.812 | 2.318x |
| prefillc_L512 | FlexAttention | 17.875 | 15.020 | 1.190x |
| prefillc_L512 | Local Neptune TileLang | 13.321 | 15.020 | 0.887x |
| prefillc_L512 | TAWA | 18.593 | 15.020 | 1.238x |
| prefillc_L512 | TileLang | 13.288 | 15.020 | 0.885x |
| prefillc_L512 | cuDNN SDPA | 13.080 | 15.020 | 0.871x |
| prefillc_L512 | PyTorch Flash | 23.144 | 15.020 | 1.541x |
| prefillc_L512 | FlashAttention-2 | 21.202 | 15.020 | 1.412x |
| prefillc_L1024 | FlexAttention | 45.254 | 34.346 | 1.318x |
| prefillc_L1024 | Local Neptune TileLang | 34.571 | 34.346 | 1.007x |
| prefillc_L1024 | TAWA | 48.415 | 34.346 | 1.410x |
| prefillc_L1024 | TileLang | 34.406 | 34.346 | 1.002x |
| prefillc_L1024 | cuDNN SDPA | 30.810 | 34.346 | 0.897x |
| prefillc_L1024 | PyTorch Flash | 59.385 | 34.346 | 1.729x |
| prefillc_L1024 | FlashAttention-2 | 54.734 | 34.346 | 1.594x |
| prefillc_L2048 | FlexAttention | 133.677 | 106.853 | 1.251x |
| prefillc_L2048 | Local Neptune TileLang | 98.689 | 106.853 | 0.924x |
| prefillc_L2048 | TAWA | 134.477 | 106.853 | 1.259x |
| prefillc_L2048 | TileLang | 98.772 | 106.853 | 0.924x |
| prefillc_L2048 | cuDNN SDPA | 84.616 | 106.853 | 0.792x |
| prefillc_L2048 | PyTorch Flash | 176.892 | 106.853 | 1.655x |
| prefillc_L2048 | FlashAttention-2 | 167.339 | 106.853 | 1.566x |
| prefillc_L4096 | FlexAttention | 428.612 | 344.299 | 1.245x |
| prefillc_L4096 | Local Neptune TileLang | 323.380 | 344.299 | 0.939x |
| prefillc_L4096 | TAWA | 410.403 | 344.299 | 1.192x |
| prefillc_L4096 | TileLang | 322.517 | 344.299 | 0.937x |
| prefillc_L4096 | cuDNN SDPA | 290.988 | 344.299 | 0.845x |
| prefillc_L4096 | PyTorch Flash | 544.872 | 344.299 | 1.583x |
| prefillc_L4096 | FlashAttention-2 | 503.755 | 344.299 | 1.463x |
| prefillc_L8192 | FlexAttention | 1473.950 | 1154.235 | 1.277x |
| prefillc_L8192 | Local Neptune TileLang | 1101.732 | 1154.235 | 0.955x |
| prefillc_L8192 | TAWA | 1383.225 | 1154.235 | 1.198x |
| prefillc_L8192 | TileLang | 1102.512 | 1154.235 | 0.955x |
| prefillc_L8192 | cuDNN SDPA | 984.541 | 1154.235 | 0.853x |
| prefillc_L8192 | PyTorch Flash | 1862.790 | 1154.235 | 1.614x |
| prefillc_L8192 | FlashAttention-2 | 1701.544 | 1154.235 | 1.474x |
| prefillc_L16384 | FlexAttention | 5774.659 | 4253.793 | 1.358x |
| prefillc_L16384 | Local Neptune TileLang | 4064.072 | 4253.793 | 0.955x |
| prefillc_L16384 | TAWA | 5087.462 | 4253.793 | 1.196x |
| prefillc_L16384 | TileLang | 4063.278 | 4253.793 | 0.955x |
| prefillc_L16384 | cuDNN SDPA | 3668.873 | 4253.793 | 0.862x |
| prefillc_L16384 | PyTorch Flash | 6865.116 | 4253.793 | 1.614x |
| prefillc_L16384 | FlashAttention-2 | 6272.842 | 4253.793 | 1.475x |
| prefillc_L32768 | FlexAttention | 22102.570 | 16312.898 | 1.355x |
| prefillc_L32768 | Local Neptune TileLang | 15716.903 | 16312.898 | 0.963x |
| prefillc_L32768 | TAWA | 19591.219 | 16312.898 | 1.201x |
| prefillc_L32768 | TileLang | 15725.679 | 16312.898 | 0.964x |
| prefillc_L32768 | cuDNN SDPA | 14260.957 | 16312.898 | 0.874x |
| prefillc_L32768 | PyTorch Flash | 26521.156 | 16312.898 | 1.626x |
| prefillc_L32768 | FlashAttention-2 | 24068.462 | 16312.898 | 1.475x |

## Runner warnings

- `decode_causal_L1024_accuracy.json`: tawa(tma_decode): 'Keyword argument mma_depth was specified but unrecognised'
- `decode_causal_L16384_accuracy.json`: tawa(tma_decode): 'Keyword argument mma_depth was specified but unrecognised'
- `decode_causal_L2048_accuracy.json`: tawa(tma_decode): 'Keyword argument mma_depth was specified but unrecognised'
- `decode_causal_L256_accuracy.json`: tawa(tma_decode): 'Keyword argument mma_depth was specified but unrecognised'
- `decode_causal_L32768_accuracy.json`: tawa(tma_decode): 'Keyword argument mma_depth was specified but unrecognised'
- `decode_causal_L4096_accuracy.json`: tawa(tma_decode): 'Keyword argument mma_depth was specified but unrecognised'
- `decode_causal_L512_accuracy.json`: tawa(tma_decode): 'Keyword argument mma_depth was specified but unrecognised'
- `decode_causal_L8192_accuracy.json`: tawa(tma_decode): 'Keyword argument mma_depth was specified but unrecognised'
