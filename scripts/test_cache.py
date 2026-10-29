#!/usr/bin/env python3
"""Test script to verify TAWA autotune cache is working."""

import sys
from pathlib import Path

# Add argus to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from argus.neptune_tawa_autotuner.cache import get_global_cache
import torch


def test_cache():
    """Test that cache has been populated with configs."""
    cache = get_global_cache()
    stats = cache.get_stats()

    print("=" * 60)
    print("TAWA Autotune Cache Test")
    print("=" * 60)
    print(f"Cache directory: {stats['cache_dir']}")
    print(f"Memory entries: {stats['memory_entries']}")
    print(f"Disk entries: {stats['disk_entries']}")
    print()

    if stats['disk_entries'] == 0:
        print("WARNING: No cache entries found!")
        print("Run: python scripts/populate_cache_from_logs.py /u/yucheny8/argus/logs")
        return False

    # Test retrieving a cached config
    print("Testing cache retrieval...")

    # Test causal config for (1, 32, 256, 128)
    cache_key = cache.generate_cache_key(
        shape=(1, 32, 256, 128),
        dtype=torch.float16,
        causal=True,
        has_alibi=False,
        is_gqa=False,
    )

    result = cache.get(cache_key)
    if result:
        print(f"✓ Found causal config for shape (1, 32, 256, 128)")
        print(f"  Config: {result.best_config}")
        print(f"  Latency: {result.best_latency:.4f}ms")
    else:
        print(f"✗ No causal config found for shape (1, 32, 256, 128)")

    # Test GQA config for (1, 64, 256, 128)
    cache_key = cache.generate_cache_key(
        shape=(1, 64, 256, 128),
        dtype=torch.float16,
        causal=False,
        has_alibi=False,
        is_gqa=True,
    )

    result = cache.get(cache_key)
    if result:
        print(f"✓ Found GQA config for shape (1, 64, 256, 128)")
        print(f"  Config: {result.best_config}")
        print(f"  Latency: {result.best_latency:.4f}ms")
    else:
        print(f"✗ No GQA config found for shape (1, 64, 256, 128)")

    print()
    print("=" * 60)
    print("Cache test complete!")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = test_cache()
    sys.exit(0 if success else 1)
