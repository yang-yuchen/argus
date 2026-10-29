"""Reuse the MTP sweep for only the two requested lengths."""

from scripts import bench_mtpc_e2e_graph_sweep as sweep

sweep.LENGTHS = (2048, 4096)
sweep.main()
