# RTX 5090 Decode and MTP-Q16 Results

Metric: Nsys sum of GPU kernel durations per cold-L2 repeat; unit: microseconds.
All rows passed correctness checks.

## Decode

| KV length | Nautilus | TileLang main GQA | GitHub Flex |
|---:|---:|---:|---:|
| 256 | 5.728 | 8.416 | 9.984 |
| 512 | 8.400 | 11.696 | 12.400 |
| 1024 | 13.456 | 18.528 | 18.016 |
| 2048 | 24.320 | 27.136 | 28.720 |
| 4096 | 44.928 | 47.936 | 51.056 |
| 8192 | 86.992 | 89.361 | 92.144 |
| 16384 | 169.184 | 172.240 | 175.360 |
| 32768 | 330.704 | 336.769 | 338.913 |

## MTP Q16

| KV length | Nautilus | Official Flex lower-right |
|---:|---:|---:|
| 256 | 4.224 | 7.232 |
| 512 | 4.384 | 8.096 |
| 1024 | 5.488 | 8.624 |
| 2048 | 6.608 | 10.624 |
| 4096 | 9.296 | 12.736 |
| 8192 | 14.128 | 18.928 |
| 16384 | 24.384 | 31.233 |
| 32768 | 45.376 | 51.984 |

## Why GitHub TileLang decode slows at long sequence lengths

The old GitHub decode kernel used one CTA per head and no split-KV decomposition. The replacement main example uses 8-way split-KV plus a combine kernel. Its original SM heuristic exceeded RTX 5090 shared memory; the measured configuration uses block_N=64 and one pipeline stage.

At L32768 the main GQA example reaches 336.8 us versus Nautilus 330.7 us. Nautilus still has a small advantage from its TMA/mbarrier pipeline and lower resource usage.

The previous WGMMA baseline was a different TileLang source and should not be treated as the GitHub TileLang result.

TileOPs commit 85e7e64 was probed separately but omitted from the tables: its dispatcher has no SM120 implementation and rejects RTX 5090 before launch.
