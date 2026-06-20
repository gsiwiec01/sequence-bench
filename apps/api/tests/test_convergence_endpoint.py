import pytest
from collections import defaultdict

def _build_convergence_response(exps, T: int, group_by: str = "architecture") -> dict:
    SAFE_ATTRS = {"architecture", "k1", "k2", "seed"}
    groups: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))

    for e in exps:
        if group_by not in SAFE_ATTRS:
            continue
        key = str(e[group_by])
        ratio = round(e["k2"] / T, 4)
        groups[key][ratio].append((e.get("convergence_epoch"), e["k2"]))

    points = []
    for group_name, by_ratio in groups.items():
        for ratio, data_list in sorted(by_ratio.items()):
            n_seeds = len(data_list)
            k2 = data_list[0][1]
            converged = [ce for ce, _ in data_list if ce is not None]
            n_converged = len(converged)

            if n_converged == 0:
                mean = std = min_val = max_val = None
            else:
                mean = sum(converged) / n_converged
                variance = sum((v - mean) ** 2 for v in converged) / max(n_converged - 1, 1)
                std = variance ** 0.5
                min_val = min(converged)
                max_val = max(converged)

            points.append({
                "k2": k2,
                "k2_ratio": ratio,
                group_by: group_name,
                "convergence_epoch_mean": mean,
                "convergence_epoch_std": std,
                "convergence_epoch_min": min_val,
                "convergence_epoch_max": max_val,
                "n_seeds": n_seeds,
                "n_converged": n_converged,
            })

    points.sort(key=lambda p: (str(p[group_by]), p["k2_ratio"]))
    return {"points": points}

def test_convergence_single_seed_all_converged():
    exps = [
        {"architecture": "lstm", "k1": 10, "k2": 30, "seed": 42, "convergence_epoch": 50},
        {"architecture": "lstm", "k1": 10, "k2": 60, "seed": 42, "convergence_epoch": 80},
    ]
    resp = _build_convergence_response(exps, T=60)
    points = resp["points"]
    assert len(points) == 2

    pt_30 = next(p for p in points if p["k2"] == 30)
    assert pt_30["convergence_epoch_mean"] == pytest.approx(50.0)
    assert pt_30["n_seeds"] == 1
    assert pt_30["n_converged"] == 1
    assert pt_30["k2_ratio"] == pytest.approx(0.5)

def test_convergence_multiple_seeds_avg():
    exps = [
        {"architecture": "lstm", "k1": 10, "k2": 30, "seed": 42, "convergence_epoch": 100},
        {"architecture": "lstm", "k1": 10, "k2": 30, "seed": 43, "convergence_epoch": 120},
        {"architecture": "lstm", "k1": 10, "k2": 30, "seed": 44, "convergence_epoch": 110},
    ]
    resp = _build_convergence_response(exps, T=30)
    points = resp["points"]
    assert len(points) == 1
    pt = points[0]
    assert pt["convergence_epoch_mean"] == pytest.approx((100 + 120 + 110) / 3)
    assert pt["n_seeds"] == 3
    assert pt["n_converged"] == 3
    assert pt["convergence_epoch_min"] == 100
    assert pt["convergence_epoch_max"] == 120

def test_convergence_partial_convergence_excluded_from_mean():
    exps = [
        {"architecture": "gru", "k1": 5, "k2": 20, "seed": 42, "convergence_epoch": 90},
        {"architecture": "gru", "k1": 5, "k2": 20, "seed": 43, "convergence_epoch": None},
        {"architecture": "gru", "k1": 5, "k2": 20, "seed": 44, "convergence_epoch": 110},
    ]
    resp = _build_convergence_response(exps, T=20)
    pt = resp["points"][0]
    assert pt["n_seeds"] == 3
    assert pt["n_converged"] == 2
    assert pt["convergence_epoch_mean"] == pytest.approx(100.0)

def test_convergence_no_convergence():
    exps = [
        {"architecture": "rnn", "k1": 5, "k2": 10, "seed": 42, "convergence_epoch": None},
    ]
    resp = _build_convergence_response(exps, T=10)
    pt = resp["points"][0]
    assert pt["n_converged"] == 0
    assert pt["convergence_epoch_mean"] is None
    assert pt["convergence_epoch_std"] is None

def test_convergence_multiple_architectures():
    T = 100
    exps = [
        {"architecture": "lstm", "k1": 10, "k2": 50, "seed": 42, "convergence_epoch": 60},
        {"architecture": "gru",  "k1": 10, "k2": 50, "seed": 42, "convergence_epoch": 80},
    ]
    resp = _build_convergence_response(exps, T=T)
    assert len(resp["points"]) == 2
    archs = {p["architecture"] for p in resp["points"]}
    assert archs == {"lstm", "gru"}

def test_convergence_std_with_two_seeds():
    exps = [
        {"architecture": "lstm", "k1": 10, "k2": 30, "seed": 42, "convergence_epoch": 100},
        {"architecture": "lstm", "k1": 10, "k2": 30, "seed": 43, "convergence_epoch": 120},
    ]
    resp = _build_convergence_response(exps, T=30)
    pt = resp["points"][0]
    mean = (100 + 120) / 2  # 110
    expected_std = (((100 - mean) ** 2 + (120 - mean) ** 2) / (2 - 1)) ** 0.5
    assert pt["convergence_epoch_std"] == pytest.approx(expected_std)

def test_convergence_empty_experiment_list():
    resp = _build_convergence_response([], T=100)
    assert resp["points"] == []
