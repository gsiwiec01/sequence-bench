from __future__ import annotations

import json

import numpy as np
from dataclasses import dataclass, field
from scipy import stats

@dataclass
class ExperimentResult:
    experiment_id: str
    architecture: str
    dataset: str
    k1: int
    k2: int
    T: int
    seed: int
    best_metric: float
    metric_name: str
    epoch_history: list[dict] = field(default_factory=list)


class ExperimentComparator:
    def degradation_curve(
            self,
            results: list[ExperimentResult],
            baseline_k2_ratio: float = 1.0,
            group_by: str = "architecture",
    ) -> dict:
        groups: dict[str, list[ExperimentResult]] = {}
        for r in results:
            key = getattr(r, group_by)
            groups.setdefault(key, []).append(r)

        output = {}
        for group_name, group_results in groups.items():
            baselines = [r for r in group_results if abs(r.k2 / r.T - baseline_k2_ratio) < 0.01]
            if not baselines:
                continue

            baseline_metric = np.mean([r.best_metric for r in baselines])

            by_ratio: dict[float, list[float]] = {}
            for r in group_results:
                ratio = round(r.k2 / r.T, 4)
                by_ratio.setdefault(ratio, []).append(r.best_metric)

            ratios = sorted(by_ratio.keys())
            deltas_mean = [np.mean(by_ratio[r]) / baseline_metric for r in ratios]
            deltas_std = [np.std(by_ratio[r]) / baseline_metric for r in ratios]

            output[group_name] = {
                "k2_ratios": ratios,
                "delta_mean": deltas_mean,
                "delta_std": deltas_std,
                "baseline_metric": baseline_metric,
            }
        return output

    def find_knee_point(self, x_values: list[float], y_values: list[float]) -> float | None:
        if len(x_values) < 3:
            return None

        x = np.array(x_values)
        y = np.array(y_values)

        x_n = (x - x.min()) / (x.ptp() + 1e-8)
        y_n = (y - y.min()) / (y.ptp() + 1e-8)

        diffs = y_n - x_n
        knee_idx = int(np.argmax(np.abs(np.diff(diffs))))

        return float(x_values[knee_idx])

    def compare_groups(
            self,
            group_a: list[ExperimentResult],
            group_b: list[ExperimentResult],
            test: str = "wilcoxon",
    ) -> dict:
        a_vals = np.array([r.best_metric for r in group_a])
        b_vals = np.array([r.best_metric for r in group_b])

        if test == "wilcoxon" and len(a_vals) == len(b_vals):
            stat, p = stats.wilcoxon(a_vals, b_vals)
        elif test == "mannwhitney":
            stat, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
        else:
            stat, p = stats.ttest_ind(a_vals, b_vals)

        pooled_std = np.sqrt((a_vals.std() ** 2 + b_vals.std() ** 2) / 2) + 1e-8
        effect_size = (a_vals.mean() - b_vals.mean()) / pooled_std

        return {
            "statistic": float(stat),
            "p_value": float(p),
            "effect_size": float(effect_size),
            "significant": bool(p < 0.05),
            "mean_a": float(a_vals.mean()),
            "mean_b": float(b_vals.mean()),
        }

    def k1_benefit(self, results: list[ExperimentResult], group_by: list[str] = None, ) -> dict:
        if group_by is None:
            group_by = ["architecture", "dataset"]

        def key_fn(r):
            return tuple(getattr(r, g) for g in group_by)

        groups: dict = {}
        for r in results:
            groups.setdefault(key_fn(r), {"k1_1": [], "k1_eq_k2": []})
            if r.k1 == 1:
                groups[key_fn(r)]["k1_1"].append(r.best_metric)
            elif r.k1 == r.k2:
                groups[key_fn(r)]["k1_eq_k2"].append(r.best_metric)

        output = {}
        for key, vals in groups.items():
            if vals["k1_1"] and vals["k1_eq_k2"]:
                benefit = np.mean(vals["k1_1"]) - np.mean(vals["k1_eq_k2"])
                output[str(key)] = {
                    "benefit": float(benefit),
                    "n_k1_1": len(vals["k1_1"]),
                    "n_k1_eq_k2": len(vals["k1_eq_k2"])
                }

        return output

    def export(self, results: list[ExperimentResult], fmt: str = "csv") -> str:
        rows = [
            {"id": r.experiment_id, "arch": r.architecture, "dataset": r.dataset,
             "k1": r.k1, "k2": r.k2, "T": r.T, "seed": r.seed,
             "metric": r.best_metric, "metric_name": r.metric_name}
            for r in results
        ]

        if fmt == "json":
            return json.dumps(rows, indent=2)

        if fmt == "csv":
            header = ",".join(rows[0].keys())
            lines = [",".join(str(v) for v in row.values()) for row in rows]
            return header + "\n" + "\n".join(lines)

        raise ValueError(f"Nieznany format: {fmt}")
