"""All figures. One visual system: thin marks, muted grid, a fixed colour order."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 1.6,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_survivorship(by_year: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(by_year["year"], by_year["pct_members_no_data"], color=SERIES[0], width=0.7)
    ax.set_title("Index members with no price data, by year (% of that year's members)")
    ax.set_ylabel("% of members")
    ax.grid(axis="x", visible=False)
    ax.set_xticks(by_year["year"][::2])
    _save(fig, path)


def plot_ic_bars(summary: pd.DataFrame, h: int, path: Path) -> None:
    """Mean rank IC per signal with a +/- 2 Newey-West standard error whisker."""
    s = summary.sort_values("mean_ic")
    se = (s["mean_ic"] / s["nw_tstat"]).abs()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(s.index, s["mean_ic"], xerr=2 * se, color=SERIES[0], height=0.6,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_title(f"Mean daily rank IC, {h}-day label (whiskers: 2 Newey-West s.e.)")
    ax.set_xlabel("mean rank IC")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_ic_decay(decay: pd.DataFrame, path: Path) -> None:
    """Small multiples: mean IC against label horizon, one panel per signal."""
    names = list(decay.index)
    fig, axes = plt.subplots(2, int(np.ceil(len(names) / 2)), figsize=(9, 3.8), sharex=True, sharey=True)
    horizons = [int(c) for c in decay.columns]
    for ax, name in zip(axes.ravel(), names):
        ax.plot(horizons, decay.loc[name].to_numpy(), color=SERIES[0], marker="o", markersize=3)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_title(name, fontsize=8, fontweight="normal")
        ax.set_xticks(horizons)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    fig.suptitle("IC decay: mean rank IC by label horizon (trading days)", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_correlation(corr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if abs(v) > 0.6 else INK)
    ax.grid(visible=False)
    ax.set_title("Average cross-sectional rank correlation between signals")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_cumulative(series: dict[str, pd.Series], title: str, path: Path) -> None:
    """Cumulative (additive) return lines with a legend and end-of-line labels."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for color, (name, r) in zip(SERIES, series.items()):
        cum = r.cumsum() * 100
        ax.plot(cum.index, cum.to_numpy(), color=color, label=name)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("cumulative return, % of gross book")
    ax.legend(loc="upper left", ncol=min(4, len(series)), fontsize=8)
    _save(fig, path)


def plot_cpcv(split_sharpes: dict[str, list[float]], path_sharpes: dict[str, list[float]], path: Path) -> None:
    """One row per model: every CPCV test-split Sharpe as a dot, path Sharpes as ticks."""
    names = list(split_sharpes)
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(names) + 1.4))
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        x = np.asarray(split_sharpes[name])
        ax.scatter(x, i + rng.uniform(-0.12, 0.12, len(x)), s=22, color=SERIES[0], alpha=0.75,
                   edgecolor=SURFACE, linewidth=0.8, label="test split" if i == 0 else None)
        ax.scatter(path_sharpes[name], np.full(len(path_sharpes[name]), i), marker="|", s=260, color=SERIES[1],
                   linewidth=1.6, label="full path" if i == 0 else None)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("annualized net Sharpe ratio, out of sample")
    ax.set_title("CPCV: distribution of out-of-sample Sharpe ratios")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=2, fontsize=8)
    ax.set_title("CPCV: distribution of out-of-sample Sharpe ratios", pad=26)
    _save(fig, path)
