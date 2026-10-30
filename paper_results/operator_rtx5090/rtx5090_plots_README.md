# RTX 5090 result plots

These are the current RTX 5090 operator plots. The source table and both
plot-input CSVs are committed; raw Nsight Systems profiles are intentionally
kept off GitHub.

- [Decode PDF](rtx5090-decode.pdf): Nautilus, TileLang main GQA, and GitHub FlexAttention.
- [Causal MTP Q16 PDF](rtx5090-causal-mtp.pdf): Nautilus and official Flex lower-right attention.
- Versioned copies are [decode](rtx5090-20261029-decode.pdf) and
  [MTP](rtx5090-20261029-causal-mtp.pdf).
- The plotted sequence lengths are 256 through 32768 in ascending order.
- Values in the CSVs and source table are **microseconds**, computed as the
  median sum of GPU kernel durations in an nsys cold-L2 repeat.
- The y-axis is `Nautilus latency / method latency`; Nautilus is 1.0 and a
  higher value means the method is faster.
- The decode TileLang curve is the SM120-safe configuration of TileLang's
  main `flash_decoding/example_gqa_decode.py` split-KV example.
- TileOPs is not plotted because its dispatcher has no SM120 implementation.

`make_plots_20261029.py` parses the Markdown table, regenerates both CSVs and
PDFs, and uses the shared `plot_from_csv_5090.py` normalization/plotting code.
The manifests record the source hash and exact file mapping.
