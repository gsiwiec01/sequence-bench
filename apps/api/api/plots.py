from __future__ import annotations

import io
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless, no GUI backend

from matplotlib.figure import Figure  # noqa: E402

GROUP_COLORS = {"lstm": "#2563eb", "gru": "#dc2626"}
FALLBACK_COLORS = ["#16a34a", "#ca8a04", "#9333ea", "#0891b2"]

_DPI = 300

def _group_color(name: str, idx: int) -> str:
    return GROUP_COLORS.get(name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def _fig_to_png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.read()


def render_degradation_png(data: dict) -> bytes:
    groups: dict = data.get("groups", {})
    meta: dict = data.get("meta", {})

    fig = Figure(figsize=(9, 6))
    ax = fig.subplots()

    all_ratios: set[float] = set()

    for idx, name in enumerate(sorted(groups.keys())):
        gd = groups[name]
        ratios = gd["k2_ratios"]
        means = gd["delta_mean"]
        stds = gd["delta_std"]
        all_ratios.update(ratios)
        color = _group_color(name, idx)
        ax.errorbar(
            ratios,
            means,
            yerr=stds,
            label=name.upper(),
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.5,
            capsize=4,
            elinewidth=1.8,
        )

    ax.axhline(y=1.0, color="#9ca3af", linestyle="--", linewidth=1.5)
    ax.text(
        0.995, 1.0, "Baseline",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=12, color="#9ca3af",
    )

    ax.set_xlabel("k₂ / T", fontsize=16, fontweight="bold")
    ax.set_ylabel("δ(k₂)", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)

    if all_ratios:
        ticks = sorted(all_ratios)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{r * 100:.0f}%" for r in ticks])

    ax.grid(True, linestyle=":", linewidth=0.8, color="#d1d5db", alpha=0.8)
    ax.legend(fontsize=13, loc="best", frameon=True)

    fig.tight_layout()
    return _fig_to_png(fig)


def render_convergence_png(data: dict) -> bytes:
    group_by: str = data.get("group_by", "architecture")
    points: list = data.get("points", [])

    by_group: dict[str, list[dict]] = {}
    for p in points:
        by_group.setdefault(str(p[group_by]), []).append(p)

    union = sorted({p["k2_ratio"] for p in points})

    fig = Figure(figsize=(9, 6))
    ax = fig.subplots()

    for idx, name in enumerate(sorted(by_group.keys())):
        color = _group_color(name, idx)
        pts = sorted(
            (p for p in by_group[name] if p.get("convergence_epoch_mean") is not None),
            key=lambda p: p["k2_ratio"],
        )
        if not pts:
            continue

        for a, b in zip(pts, pts[1:]):
            i0, i1 = union.index(a["k2_ratio"]), union.index(b["k2_ratio"])
            style = "-" if i1 == i0 + 1 else (0, (5, 4))
            ax.plot(
                [a["k2_ratio"], b["k2_ratio"]],
                [a["convergence_epoch_mean"], b["convergence_epoch_mean"]],
                color=color, linestyle=style, linewidth=2.5, zorder=1,
            )

        for p in pts:
            partial = p["n_converged"] < p["n_seeds"]
            std = p.get("convergence_epoch_std") if p["n_converged"] > 1 else None
            ax.errorbar(
                p["k2_ratio"], p["convergence_epoch_mean"], yerr=std,
                color=color, capsize=4, elinewidth=1.8, zorder=2,
                marker="o", markersize=8,
                markerfacecolor="white" if partial else color,
                markeredgecolor=color, markeredgewidth=2,
            )

        ax.plot([], [], color=color, marker="o", markersize=8, linewidth=2.5, label=name.upper())

    ax.set_xlabel("k₂ / T", fontsize=16, fontweight="bold")
    ax.set_ylabel("epoka zbieżności", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)

    if union:
        ax.set_xticks(union)
        ax.set_xticklabels([f"{r * 100:.0f}%" for r in union])

    ax.grid(True, linestyle=":", linewidth=0.8, color="#d1d5db", alpha=0.8)
    if by_group:
        ax.legend(fontsize=13, loc="best", frameon=True)

    fig.tight_layout()
    return _fig_to_png(fig)


def _param_color(name: str) -> str:
    if "decoder" in name:
        return "#000000"

    if "_hh" in name:
        return "#0072B2"

    if "_ih" in name:
        return "#D55E00"

    return "#6b7280"


def render_gradient_trend_png(data: dict) -> bytes:
    epochs: list = data.get("epochs", [])
    params: dict = data.get("params", {})

    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()

    for name in sorted(params.keys()):
        vals = params[name]
        xs = [e for e, v in zip(epochs, vals) if v is not None and v > 0]
        ys = [v for v in vals if v is not None and v > 0]
        if not xs:
            continue
        ax.plot(
            xs, ys,
            color=_param_color(name),
            linestyle=(0, (4, 2)) if "bias" in name else "-",
            linewidth=2.2, label=name,
        )

    ax.set_yscale("log")
    ax.set_xlabel("epoka", fontsize=16, fontweight="bold")
    ax.set_ylabel("norma gradientu", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="major", linestyle=":", linewidth=0.9, color="#cbd5e1", alpha=0.9)

    if params:
        ax.legend(fontsize=11, loc="best", bbox_to_anchor=(1.01, 0.5), frameon=True)

    fig.tight_layout()
    return _fig_to_png(fig)


def render_loss_png(metrics: list[dict]) -> bytes:
    fig = Figure(figsize=(9, 6))
    ax = fig.subplots()

    epochs = [m["epoch"] for m in metrics]
    train = [m.get("train_loss") for m in metrics]
    val_xy = [(m["epoch"], m["val_loss"]) for m in metrics if m.get("val_loss") is not None]

    if any(v is not None for v in train):
        ax.plot(epochs, train, color="#2563eb", linewidth=2.5, label="Train loss")
    if val_xy:
        ax.plot([x for x, _ in val_xy], [y for _, y in val_xy],
                color="#dc2626", linewidth=2.5, label="Val loss")

    ax.set_xlabel("epoka", fontsize=16, fontweight="bold")
    ax.set_ylabel("strata", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle=":", linewidth=0.8, color="#d1d5db", alpha=0.8)
    ax.legend(fontsize=13, loc="best", frameon=True)

    fig.tight_layout()
    return _fig_to_png(fig)


def render_metric_png(series: list[dict], name: str, color: str = "#7c3aed") -> bytes:
    fig = Figure(figsize=(9, 6))
    ax = fig.subplots()

    xs = [p["epoch"] for p in series]
    ys = [p["value"] for p in series]
    if xs:
        ax.plot(xs, ys, color=color, linewidth=2.5, label=name)

    ax.set_xlabel("epoka", fontsize=16, fontweight="bold")
    ax.set_ylabel(name, fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle=":", linewidth=0.8, color="#d1d5db", alpha=0.8)

    fig.tight_layout()
    return _fig_to_png(fig)


def render_gradient_heatmap_png(
    data: dict,
    epoch: int | None = None,
    label: str | None = None,
) -> bytes:
    params = list(data.keys())
    if not params:
        fig = Figure(figsize=(6, 2))
        ax = fig.subplots()
        ax.text(0.5, 0.5, "Brak danych gradientów", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        return _fig_to_png(fig)

    max_steps = max((len(data[p]) for p in params), default=1)

    matrix = np.full((len(params), max_steps), np.nan)
    for i, p in enumerate(params):
        vals = np.asarray(data[p], dtype=float)
        matrix[i, : len(vals)] = vals

    with np.errstate(divide="ignore", invalid="ignore"):
        log_mat = np.where((matrix > 0) & np.isfinite(matrix), np.log10(matrix), np.nan)

    finite = log_mat[np.isfinite(log_mat)]
    if finite.size == 0:
        lv_min, lv_max = -8.0, 0.0
    else:
        lv_min = float(finite.min())
        lv_max = float(finite.max())
    if lv_max <= lv_min:
        lv_max = lv_min + 1.0

    cell_h_in = 0.28
    fig_h = max(3.5, len(params) * cell_h_in + 2.0)
    fig_w = max(8.5, min(18.0, max_steps * 0.06 + 4.0))

    fig = Figure(figsize=(fig_w, fig_h))
    ax = fig.subplots()

    masked = np.ma.masked_invalid(log_mat)
    im = ax.imshow(masked, aspect="auto", cmap="viridis", vmin=lv_min, vmax=lv_max,
                   interpolation="nearest", origin="upper")

    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("norma gradientu (log)", fontsize=11)

    lo = math.ceil(lv_min)
    hi = math.floor(lv_max)
    decade_ticks = list(range(lo, hi + 1))
    if not decade_ticks:
        decade_ticks = [round((lv_min + lv_max) / 2, 1)]
    cbar.set_ticks(decade_ticks)
    cbar.set_ticklabels([f"$10^{{{v}}}$" for v in decade_ticks], fontsize=9)

    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params, fontsize=max(6, min(9, int(180 / len(params)))))

    tick_every = max(10, (max_steps // 15 // 10) * 10)
    xticks = list(range(0, max_steps, tick_every))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(s) for s in xticks], fontsize=9)

    ax.set_xlabel("krok optymalizacji", fontsize=13, fontweight="bold")
    title = label or (f"Epoka {epoch}" if epoch is not None else "Heatmapa norm gradientów")
    ax.set_title(title, fontsize=13)

    fig.tight_layout()
    return _fig_to_png(fig)


def render_surface_png(data: dict) -> bytes:
    a = np.asarray(data["x_values"], dtype=float)
    b = np.asarray(data["y_values"], dtype=float)
    grid = np.asarray(data["loss_grid"], dtype=float)
    a_traj = np.asarray(data.get("a_traj") or [], dtype=float)
    b_traj = np.asarray(data.get("b_traj") or [], dtype=float)

    fig = Figure(figsize=(8.5, 6.5))
    ax = fig.subplots()

    A, B = np.meshgrid(a, b)
    cf = ax.contourf(A, B, grid, levels=30, cmap="viridis")
    cs = ax.contour(A, B, grid, levels=12, colors="black", linewidths=0.4, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.3g")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("val_loss", fontsize=13)

    if a_traj.size and b_traj.size:
        ax.plot(a_traj, b_traj, "-o", color="#dc2626", ms=4, lw=1.8, label="trajektoria wag", zorder=4)
        ax.scatter([a_traj[0]], [b_traj[0]], s=80, facecolor="white",
                   edgecolor="#dc2626", linewidths=1.8, zorder=5, label="start (w₀)")
        ax.scatter([a_traj[-1]], [b_traj[-1]], marker="*", s=240, color="gold",
                   edgecolor="black", linewidths=1.0, zorder=5, label="koniec (w_E)")

    ax.set_xlabel("PC1", fontsize=16, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=11, loc="best", frameon=True)

    fig.tight_layout()
    return _fig_to_png(fig)
