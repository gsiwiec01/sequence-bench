from ml_engine.analysis import ExperimentComparator, ExperimentResult


def make_results(arch, k2_ratios, T=100, base_metric=0.9, noise=0.02):
    results = []
    for ratio in k2_ratios:
        for seed in range(3):
            import random
            results.append(ExperimentResult(
                experiment_id=f"{arch}_{ratio}_{seed}",
                architecture=arch,
                dataset="copy_task",
                k1=1, k2=int(ratio * T), T=T, seed=seed,
                best_metric=base_metric * ratio + random.gauss(0, noise),
                metric_name="accuracy",
            ))

    return results


def test_degradation_curve_returns_all_groups():
    comp = ExperimentComparator()
    results = make_results("lstm", [0.05, 0.1, 0.25, 0.5, 1.0]) + \
              make_results("gru", [0.05, 0.1, 0.25, 0.5, 1.0])
    curve = comp.degradation_curve(results, group_by="architecture")

    assert "lstm" in curve and "gru" in curve
    assert len(curve["lstm"]["k2_ratios"]) == 5


def test_compare_groups_returns_p_value():
    comp = ExperimentComparator()
    group_a = make_results("lstm", [1.0])
    group_b = make_results("gru", [1.0])
    result = comp.compare_groups(group_a, group_b, test="mannwhitney")

    assert 0.0 <= result["p_value"] <= 1.0
    assert "effect_size" in result


def test_export_csv():
    comp = ExperimentComparator()
    results = make_results("lstm", [1.0], T=50)
    csv = comp.export(results, fmt="csv")
    lines = csv.strip().split("\n")

    assert len(lines) >= 2
    assert "arch" in lines[0]
