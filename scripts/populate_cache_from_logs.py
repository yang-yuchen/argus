#!/usr/bin/env python3
"""Script to populate TAWA autotune cache from existing log files.

This script parses log files containing autotuning results
and populates the cache so that future runs can skip autotuning.
"""

import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import ast

# Add argus to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from argus.neptune_tawa_autotuner.cache import AutotuneCache, AutotuneResult
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_config_string(config_str: str) -> Dict:
    """Parse config dictionary string from log."""
    try:
        # Use ast.literal_eval to safely parse the dictionary string
        return ast.literal_eval(config_str)
    except Exception as e:
        logger.error(f"Failed to parse config: {config_str}, error: {e}")
        return {}


def parse_latency(latency_str: str) -> float:
    """Parse latency value from string like '4.5524ms'."""
    try:
        return float(latency_str.replace('ms', ''))
    except Exception as e:
        logger.error(f"Failed to parse latency: {latency_str}, error: {e}")
        return 0.0


def extract_shape_from_operator_line(operator_line: str) -> Optional[Tuple[int, ...]]:
    """Extract shape from operator line.

    Example: 'prefill_causal-1,32,32,256,256,128' -> (1, 32, 256, 128)
    """
    try:
        # Extract the shape part from the line
        match = re.search(r'prefill_(?:causal|gqa|global)-(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)', operator_line)
        if match:
            b, h, kv_h, sq, sk, d = map(int, match.groups())
            # Return (batch, heads, seq_len, head_dim)
            return (b, h, sq, d)
        return None
    except Exception as e:
        logger.debug(f"Failed to extract shape from: {operator_line}, error: {e}")
        return None


def parse_log_file(log_path: Path) -> List[Tuple[str, Dict, float, Optional[Tuple]]]:
    """Parse a log file and extract best configurations.

    Returns:
        List of tuples (config_type, config_dict, latency, shape_tuple)
        where config_type is one of: 'causal', 'gqa', 'global'
    """
    results = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Pattern to match best config lines (excluding FP8 and BF16)
    best_config_pattern = re.compile(
        r'Best (causal|GQA|global) config: (\{.*?\}), latency: ([\d.]+ms)'
    )

    # Look for shape context before best config lines
    current_shape = None

    for i, line in enumerate(lines):
        # Try to extract shape from profiling lines
        if 'Profiling' in line or 'prefill_' in line:
            shape = extract_shape_from_operator_line(line)
            if shape:
                current_shape = shape

        # Match best config lines
        match = best_config_pattern.search(line)
        if match:
            config_type = match.group(1).lower()
            if config_type == 'gqa':
                config_type = 'gqa'

            config_str = match.group(2)
            latency_str = match.group(3)

            config_dict = parse_config_string(config_str)
            latency = parse_latency(latency_str)

            if config_dict and latency > 0:
                # Try to find shape from nearby context if we don't have one
                if current_shape is None:
                    # Look backward for shape information
                    for j in range(max(0, i-50), i):
                        shape = extract_shape_from_operator_line(lines[j])
                        if shape:
                            current_shape = shape
                            break

                results.append((config_type, config_dict, latency, current_shape))
                logger.info(f"Found {config_type} config: {config_dict}, latency: {latency}ms, shape: {current_shape}")

    return results


def populate_cache_from_logs(log_dir: Path, cache_dir: Optional[Path] = None):
    """Populate cache from all log files in directory.

    Args:
        log_dir: Directory containing log files
        cache_dir: Cache directory (default: ~/.cache/tawa_autotune)
    """
    # Initialize cache
    cache = AutotuneCache(cache_dir)

    # Find all log files
    log_files = list(log_dir.glob("profile_*.log"))
    logger.info(f"Found {len(log_files)} log files in {log_dir}")

    total_configs = 0
    configs_by_type = {'causal': 0, 'gqa': 0, 'global': 0}

    for log_file in log_files:
        logger.info(f"\nProcessing {log_file.name}...")

        try:
            configs = parse_log_file(log_file)

            for config_type, config_dict, latency, shape in configs:
                if shape is None:
                    logger.warning(f"Skipping config without shape: {config_type}")
                    continue

                # Determine dtype, causal, is_gqa from config_type
                dtype = torch.float16  # Default from logs
                is_gqa = (config_type == 'gqa')
                causal = (config_type == 'causal')

                # Generate cache key
                cache_key = cache.generate_cache_key(
                    shape=shape,
                    dtype=dtype,
                    causal=causal,
                    has_alibi=False,
                    is_gqa=is_gqa,
                )

                # Create AutotuneResult
                result = AutotuneResult(
                    best_config=config_dict,
                    best_latency=latency,
                    shape_signature=shape,
                    dtype=str(dtype),
                    causal=causal,
                    has_alibi=False,
                    is_gqa=is_gqa,
                )

                # Store in cache
                cache.set(cache_key, result, save_to_disk=True)
                total_configs += 1
                configs_by_type[config_type] += 1

                logger.debug(f"Cached {config_type} config for shape {shape}")

        except Exception as e:
            logger.error(f"Error processing {log_file.name}: {e}")
            continue

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Cache Population Summary:")
    logger.info(f"{'='*60}")
    logger.info(f"Total configs cached: {total_configs}")
    logger.info(f"  - Causal: {configs_by_type['causal']}")
    logger.info(f"  - GQA: {configs_by_type['gqa']}")
    logger.info(f"  - Global: {configs_by_type['global']}")
    logger.info(f"\nCache stats: {cache.get_stats()}")
    logger.info(f"{'='*60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Populate TAWA autotune cache from log files"
    )
    parser.add_argument(
        'log_dir',
        type=Path,
        help='Directory containing profile_*.log files'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help='Cache directory (default: ~/.cache/tawa_autotune)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.log_dir.exists():
        logger.error(f"Log directory not found: {args.log_dir}")
        sys.exit(1)

    populate_cache_from_logs(args.log_dir, args.cache_dir)


if __name__ == '__main__':
    main()
