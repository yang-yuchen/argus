# GH200 FP16 tensor-compiler baselines, PF_GLOBAL B1 H32 Q=KV=512 D=128

Latest baseline mean latency is from 50 cold-L2 nsys repeats in the current environment. Nautilus is the compatible-runtime measurement from the earlier run; TVM is not available in the current latest environment.

| Method | Historical PDF us | Latest us | Nautilus/latest | Latest RMSE | Status |
|---|---:|---:|---:|---:|---|
| Nautilus | 13.598 | 13.172 | 1.000x | — | legacy FP32 SDPA check |
| FlexAttn | 17.788 | 16.721 | 1.269x | 1.985e-5 | FP64 check passed |
| Helion | 16.847 | 78.100 | 5.929x | 1.985e-5 | FP64 check passed |
| OpenAI Triton | 16.386 | 15.953 | 1.211x | 1.985e-5 | FP64 check passed |
| TAWA | 14.300 | 18.007 | 1.367x | 1.995e-5 | FP64 check passed |
| TVM | 88.350 | — | — | — | not measured in latest env |
| TileLang | 16.240 | 13.356 | 1.014x | 1.995e-5 | FP64 check passed |
| Tri-Dao Triton | 21.850 | 19.620 | 1.490x | 2.080e-5 | FP64 check passed |

The historical values are from `nautilus-figures-repro/data/datasheet_new.csv`, not fresh measurements. OpenAI Triton, Tri-Dao Triton, and TAWA use current Argus runner implementations; FlexAttention/PyTorch/TileLang/Helion/FlashInfer use current package environments. The benchmark was run on GH200, not H100.
