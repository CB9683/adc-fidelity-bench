"""Descriptive toy-pilot plots with anatomy-level aggregation and failure counts."""

from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _curve(rows, metric, method, snr, radius=None):
    # First average conditions within each anatomy/budget, then anatomies equally.
    groups = defaultdict(list)
    for row in rows:
        if row['method'] == method and row['snr'] == snr and (radius is None or row['focal_radius_mm'] == radius):
            if row[metric] is not None:
                groups[(row['anatomy_id'], row['budget_per_b'])].append(row[metric])
    budgets = sorted({row['budget_per_b'] for row in rows})
    means = []
    for budget in budgets:
        values = [np.mean(values) for (_, b), values in groups.items() if b == budget]
        means.append(np.mean(values) if values else np.nan)
    return budgets, means


def save_plots(anatomy_rows, output):
    snrs = sorted({row['snr'] for row in anatomy_rows})
    methods = sorted({row['method'] for row in anatomy_rows})
    radii = sorted({row['focal_radius_mm'] for row in anatomy_rows})
    specs = [('rmse', 'ADC RMSE (mm²/s)', 'error_by_budget.png'),
             ('recovery_ratio', 'Signed focal recovery ratio', 'focal_recovery.png'),
             ('invalid_adc_fraction', 'Invalid ADC fraction', 'invalid_by_budget.png')]
    for metric, label, filename in specs:
        fig, axes = plt.subplots(1, len(snrs), figsize=(5 * len(snrs), 4.2), squeeze=False)
        for ax, snr in zip(axes[0], snrs):
            for method in methods:
                for radius in radii if metric == 'recovery_ratio' else [None]:
                    x, y = _curve(anatomy_rows, metric, method, snr, radius)
                    legend = method + (f', r={radius:g} mm' if radius is not None else '')
                    ax.plot(x, y, marker='o', label=legend)
            if metric == 'recovery_ratio':
                ax.axhline(1, color='gray', linewidth=0.8, linestyle='--')
            ax.set(title=f'Reference SNR {snr:g}', xlabel='Repetitions per b-value', ylabel=label)
            ax.set_xticks(sorted({row['budget_per_b'] for row in anatomy_rows}))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        subtitle = ('Defined full-focal cases only; inspect failure columns' if metric == 'recovery_ratio'
                    else 'Finite ADC estimates only; inspect invalid fractions' if metric == 'rmse'
                    else 'All requested tissue voxels retained in denominator')
        fig.suptitle(f'Original numerical phantoms · {subtitle}\nEqual anatomy weight; conditions averaged; no uncertainty interval', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
