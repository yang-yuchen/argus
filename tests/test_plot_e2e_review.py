import matplotlib.pyplot as plt
import pandas as pd
import pytest

from scripts import plot_e2e_review as paper


def test_latency_to_throughput_and_original_grouped_bar_style(tmp_path):
    path = tmp_path / 'measurements.csv'
    pd.DataFrame({'kv_length': [256, 256], 'implementation': ['nautilus', 'tileops'],
                  'mean_ms': [10.0, 20.0]}).to_csv(path, index=False)
    data = paper.load_data(path, query_length=1)
    assert data['neptune_tawa'].iloc[0] == pytest.approx(1.0200547328)
    assert data['tilelang'].iloc[0] == pytest.approx(0.5100273664)
    fig, ax = paper.plot(data, tmp_path / 'preview', 'Decode')
    try:
        assert len(ax.containers) == 2
        assert ax.get_ylabel() == 'Throughput (TFLOPs/s)'
        assert ax.get_ylim()[0] == 0
        assert len(ax.lines) == 0
    finally:
        plt.close(fig)


def test_missing_baseline_lengths_are_rejected(tmp_path):
    path = tmp_path / 'partial.csv'
    pd.DataFrame({'kv_length': [256, 256, 512],
                  'implementation': ['nautilus', 'tileops', 'nautilus'],
                  'mean_ms': [10.0, 20.0, 10.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Missing'):
        paper.load_data(path, query_length=1)
