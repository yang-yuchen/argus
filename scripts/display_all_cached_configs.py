#!/usr/bin/env python3
"""Display all cached configurations in a readable format."""

import sys
import json
from pathlib import Path

# Add argus to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from argus.neptune_tawa_autotuner.cache import get_global_cache, AutotuneResult


def display_all_configs():
    """Display all cached configurations."""
    cache = get_global_cache()
    cache_dir = Path(cache.cache_dir)

    # Find all cache subdirectories
    cache_entries = [d for d in cache_dir.iterdir() if d.is_dir()]

    print("=" * 80)
    print(f"All Cached TAWA Configurations ({len(cache_entries)} total)")
    print("=" * 80)
    print()

    # Group by attention type
    causal_configs = []
    gqa_configs = []
    global_configs = []

    for entry_dir in sorted(cache_entries):
        cache_file = entry_dir / "autotune_result.json"
        if not cache_file.exists():
            continue

        with open(cache_file, 'r') as f:
            data = json.load(f)

        result = AutotuneResult.from_dict(data)

        # Categorize by type
        # Check if shape has expanded heads (GQA pattern) by looking at the data
        is_gqa = data.get('is_gqa', False) if 'is_gqa' in data else False

        if is_gqa:
            gqa_configs.append((entry_dir.name[:16], result))
        elif result.causal:
            causal_configs.append((entry_dir.name[:16], result))
        else:
            global_configs.append((entry_dir.name[:16], result))

    # Display Causal configs
    if causal_configs:
        print("CAUSAL ATTENTION CONFIGS")
        print("-" * 80)
        for cache_key, result in causal_configs:
            print(f"Cache Key: {cache_key}...")
            print(f"  Shape: {result.shape_signature}")
            print(f"  Dtype: {result.dtype}")
            print(f"  Latency: {result.best_latency:.4f}ms")
            print(f"  Config: {result.best_config}")
            print()

    # Display GQA configs
    if gqa_configs:
        print("GQA ATTENTION CONFIGS")
        print("-" * 80)
        for cache_key, result in gqa_configs:
            print(f"Cache Key: {cache_key}...")
            print(f"  Shape: {result.shape_signature}")
            print(f"  Dtype: {result.dtype}")
            print(f"  Latency: {result.best_latency:.4f}ms")
            print(f"  Config: {result.best_config}")
            print()

    # Display Global configs
    if global_configs:
        print("GLOBAL ATTENTION CONFIGS")
        print("-" * 80)
        for cache_key, result in global_configs:
            print(f"Cache Key: {cache_key}...")
            print(f"  Shape: {result.shape_signature}")
            print(f"  Dtype: {result.dtype}")
            print(f"  Latency: {result.best_latency:.4f}ms")
            print(f"  Config: {result.best_config}")
            print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Total configs: {len(cache_entries)}")
    print(f"  - Causal: {len(causal_configs)}")
    print(f"  - GQA: {len(gqa_configs)}")
    print(f"  - Global: {len(global_configs)}")
    print("=" * 80)


if __name__ == '__main__':
    display_all_configs()
