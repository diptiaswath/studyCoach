"""
plot_ablations.py
─────────────────
Loads all eval result JSONs from ./eval_results/ (including subdirectories)
and produces comparison figures for the course report.

Generates:
  1. metrics_comparison.png   — ROUGE-L / METEOR / BERTScore across all runs
  2. routing_entropy.png      — router entropy per MoLoRA run
  3. expert_load_heatmap.png  — per-expert load by figure type (best MoLoRA run)
  4. pareto_frontier.png      — accuracy vs. trainable params
  5. temperature_ablation.png — entropy + eval metric vs router_temp_init (ablation 4)

Usage
─────
  python plot_ablations.py [--eval_dir ./eval_results] [--plot_dir ./plots]
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


METRICS = ["ROUGE-L", "METEOR", "BERTScore-F1"]
_CMAP   = cm.get_cmap("tab10")


def _colour(i: int):
    return _CMAP(i % 10)


def _short_label(name: str) -> str:
    return name.replace("ablation-", "").replace("_", "\n").replace("-", "\n", 2)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_results(eval_dir: Path) -> dict:
    """
    Load all result JSONs from eval_dir.
    Results are flat files named <ablation>.json, e.g.:
      eval_results/molora-K4-r16.json
      eval_results/baseline-zeroshot.json
    Also handles the legacy subdirectory layout for backward compatibility.
    The "ablation" field in the JSON is used as the key when present.
    """
    records = {}

    def _ingest(path: Path, fallback_key: str):
        with open(path) as f:
            raw = json.load(f)
        # Use the "ablation" field as the key if present, else fallback
        key = raw.get("ablation") or fallback_key
        records[key] = {
            "ablation":         key,
            "scores":           raw.get("scores", {}),
            "routing":          raw.get("routing", {}),
            "by_fig_type":      raw.get("by_fig_type", {}),
            "trainable_params": raw.get("trainable_params"),
            "is_molora":        "routing" in raw and bool(raw.get("routing")),
            "n_examples":       raw.get("n_examples", len(raw.get("predictions", []))),
        }

    # Flat layout: eval_results/<ablation>.json (current)
    for path in sorted(eval_dir.glob("*.json")):
        _ingest(path, path.stem)

    # Legacy subdirectory layout: eval_results/<run_name>/<file>.json
    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for path in sorted(subdir.glob("*.json")):
            _ingest(path, subdir.name)
            break

    return records


# ─────────────────────────────────────────────
# Plot 1: metrics comparison
# ─────────────────────────────────────────────

def plot_metrics_comparison(records: dict, plot_dir: Path):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5), sharey=False)
    if len(METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        values, labels, colours = [], [], []
        for i, (name, data) in enumerate(records.items()):
            v = data["scores"].get(metric)
            if v is not None:
                values.append(v)
                labels.append(_short_label(name))
                colours.append(_colour(i))
        if not values:
            ax.set_title(f"{metric}\n(no data)")
            continue
        bars = ax.bar(range(len(values)), values, color=colours,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    fig.suptitle("SPIQA+ Test-A: Study Coach Assessment\n"
                 "MoLoRA vs. Baselines (eval on 108 test examples)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = plot_dir / "metrics_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────
# Plot 2: routing entropy scatter
# ─────────────────────────────────────────────

def plot_routing_entropy(records: dict, plot_dir: Path):
    molora = {k: v for k, v in records.items() if v["is_molora"]}
    if not molora:
        print("No MoLoRA routing data — skipping entropy plot")
        return

    names, entropies, colours = [], [], []
    for i, (name, data) in enumerate(molora.items()):
        e = data["routing"].get("router_entropy")
        if e is not None:
            names.append(_short_label(name))
            entropies.append(e)
            colours.append(_colour(i))
    if not names:
        print("router_entropy not found in routing data — skipping")
        return

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.9), 4))
    ax.scatter(range(len(names)), entropies, s=100, color=colours, zorder=3,
               edgecolors="white", linewidths=0.5)
    for x, y in enumerate(entropies):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)

    ax.axhline(np.log(4), color="steelblue", linestyle="--", alpha=0.6,
               label=f"Max entropy K=4 ({np.log(4):.3f} nats)")
    ax.axhline(np.log(2), color="orange",    linestyle=":",  alpha=0.6,
               label=f"Max entropy K=2 ({np.log(2):.3f} nats)")
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.4,
               label="Collapse threshold (~0.5)")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Router Entropy (nats)")
    ax.set_ylim(0, np.log(4) * 1.25)
    ax.set_title("Routing Entropy by Configuration\n(higher = more balanced)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = plot_dir / "routing_entropy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────
# Plot 3: expert load heatmap
# ─────────────────────────────────────────────

def plot_expert_load_heatmap(records: dict, plot_dir: Path):
    molora = {k: v for k, v in records.items() if v["is_molora"]}
    if not molora:
        return

    best_name, best_data, best_score = None, None, -1
    for name, data in molora.items():
        rbt = data["routing"].get("routing_by_fig_type", {})
        if not rbt:
            continue
        score = data["scores"].get("BERTScore-F1", 0)
        if score > best_score:
            best_score, best_name, best_data = score, name, data

    if best_name is None:
        print("No per-figure-type routing data found — skipping heatmap")
        return

    routing_by_type = best_data["routing"]["routing_by_fig_type"]
    fig_types = sorted(routing_by_type.keys())
    K = max(
        sum(1 for k in v if re.match(r"expert_\d+$", k))
        for v in routing_by_type.values()
    )
    if K == 0:
        return

    matrix = np.zeros((len(fig_types), K))
    for i, ft in enumerate(fig_types):
        for j in range(K):
            matrix[i, j] = routing_by_type[ft].get(f"expert_{j}", 0.0)

    fig, ax = plt.subplots(figsize=(max(6, K * 1.5), max(4, len(fig_types) * 0.7)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=matrix.max())
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"Expert {j}" for j in range(K)], fontweight="bold")
    ax.set_yticks(range(len(fig_types)))
    ax.set_yticklabels(fig_types, fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean routing weight")
    for i in range(len(fig_types)):
        for j in range(K):
            val = matrix[i, j]
            tc  = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=tc)
        dom = int(np.argmax(matrix[i]))
        ax.add_patch(plt.Rectangle((dom - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="black", lw=2))
    ax.set_title(f"Expert Routing by Figure Type\n({_short_label(best_name)})",
                 fontweight="bold")
    plt.tight_layout()
    out = plot_dir / "expert_load_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────
# Plot 4: Pareto frontier
# ─────────────────────────────────────────────

def plot_pareto(records: dict, plot_dir: Path):
    xs, ys, labels, colours = [], [], [], []

    for i, (name, data) in enumerate(records.items()):
        score = data["scores"].get("BERTScore-F1")
        if score is None:
            continue

        params = data.get("trainable_params")
        if params is None:
            mk = re.search(r"K(\d+)", name)
            mr = re.search(r"[^a-z]r(\d+)", name)
            K  = int(mk.group(1)) if mk else None
            r  = int(mr.group(1)) if mr else None
            # Qwen3-VL-8B: hidden=4096, 28 LLM layers, 4 target modules
            if "lora-r" in name and r:
                params = 2 * r * 4096 * 28 * 4
            elif "molora" in name and K and r:
                params = K * 2 * r * 4096 * 28 * 4 + 4096 * 128 + 128 * K
            elif "zeroshot" in name or "baseline" in name:
                params = 0

        if params is None:
            continue

        xs.append(params / 1e6)
        ys.append(score)
        labels.append(_short_label(name))
        colours.append(_colour(i))

    if not xs:
        print("Insufficient data for Pareto plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for x, y, label, c in zip(xs, ys, labels, colours):
        ax.scatter(x, y, s=120, color=c, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 3), fontsize=7)
    ax.set_xlabel("Trainable parameters (M)")
    ax.set_ylabel("BERTScore-F1")
    ax.set_title("Accuracy vs. Parameter Count", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = plot_dir / "pareto_frontier.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────
# Plot 5: temperature ablation (new)
# ─────────────────────────────────────────────

def plot_temperature_ablation(records: dict, plot_dir: Path):
    """
    Dual-axis line plot: router entropy + BERTScore-F1 vs router_temp_init.
    temp=1.0 is the no-annealing (collapse) condition.
    """
    temp_runs = {k: v for k, v in records.items() if re.search(r"temp[\d.]+", k)}
    if not temp_runs:
        print("No temperature ablation runs found — skipping temperature plot")
        return

    def _temp(name):
        m = re.search(r"temp([\d.]+)", name)
        return float(m.group(1)) if m else None

    rows = sorted(
        [(t, data) for name, data in temp_runs.items()
         if (t := _temp(name)) is not None],
        key=lambda x: x[0]
    )
    if not rows:
        return

    temps     = [r[0] for r in rows]
    entropies = [r[1]["routing"].get("router_entropy") for r in rows]
    scores    = [r[1]["scores"].get("BERTScore-F1")    for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    l1, = ax1.plot(temps, entropies, "o-", color="steelblue", linewidth=2,
                   markersize=9, label="Router entropy (nats)", zorder=3)
    ax1.axhline(np.log(4), color="steelblue", linestyle="--", alpha=0.35,
                label=f"Max entropy ({np.log(4):.2f} nats)")
    ax1.axhline(0.5, color="red", linestyle=":", alpha=0.5,
                label="Collapse threshold (~0.5)")
    ax1.set_xlabel("router_temp_init", fontweight="bold")
    ax1.set_ylabel("Router Entropy (nats)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_ylim(0, np.log(4) * 1.3)

    valid = [(t, s) for t, s in zip(temps, scores) if s is not None]
    lines = [l1]
    if valid:
        ts, ss = zip(*valid)
        l2, = ax2.plot(ts, ss, "s--", color="darkorange", linewidth=2,
                       markersize=9, label="BERTScore-F1", zorder=3)
        ax2.set_ylabel("BERTScore-F1", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")
        margin = (max(ss) - min(ss)) * 0.5 if max(ss) > min(ss) else 0.02
        ax2.set_ylim(min(ss) - margin, max(ss) + margin)
        lines.append(l2)

    ax1.set_xticks(temps)
    ax1.set_title("Effect of Router Temperature Annealing\n"
                  "(temp=1.0 = no annealing = collapse condition)",
                  fontweight="bold")
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    out = plot_dir / "temperature_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────
# Plot 6: architecture comparison (priority ablation set)
# ─────────────────────────────────────────────

# Maps substrings in run names → display labels and group assignments
_RUN_CATALOGUE = [
    # (name_substring,          display_label,        group,        marker, colour_idx)
    ("baseline-zeroshot",       "Zero-shot",           "baseline",   "D",    7),
    ("baseline-lora-r16",       "LoRA r=16",           "lora",       "s",    1),
    ("baseline-lora-r64",       "LoRA r=64",           "lora",       "s",    2),
    ("molora-K2-r16",           "MoLoRA K=2 r=16",     "molora",     "o",    3),
    ("molora-K4-r8",            "MoLoRA K=4 r=8",      "molora",     "o",    4),
    ("molora-K4-r16",           "MoLoRA K=4 r=16\n(primary, 8k)", "molora", "o", 0),
    # Stage 2: merger-unfrozen variants
    ("lora-r64-merger",         "LoRA r=64\n+merger",          "lora",   "s",  6),
    ("molora-K4-r16-merger",    "MoLoRA K=4 r=16\n+merger",    "molora", "^",  5),
]

# Trainable param counts (millions) for each run
# Qwen3-VL-8B: 28 LLM layers, hidden=4096, 4 target modules (q/v/o/gate)
_PARAMS_M = {
    "baseline-zeroshot":  0.0,
    "baseline-lora-r16":  round(2 * 16 * 4096 * 28 * 4 / 1e6, 1),   # ~14.7M
    "baseline-lora-r64":  round(2 * 64 * 4096 * 28 * 4 / 1e6, 1),   # ~58.7M
    "molora-K2-r16":      round((2 * 2 * 16 * 4096 * 28 * 4 + 4096*128 + 128*2) / 1e6, 1),
    "molora-K4-r8":       round((4 * 2 * 8  * 4096 * 28 * 4 + 4096*128 + 128*4) / 1e6, 1),
    "molora-K4-r16":      round((4 * 2 * 16 * 4096 * 28 * 4 + 4096*128 + 128*4) / 1e6, 1),
    # merger-tuned: same adapter params + merger (~1280->4096, ~5.2M extra)
    "lora-r64-merger":        round((2 * 64 * 4096 * 28 * 4 + 1280*4096 + 4096) / 1e6, 1),
    "molora-K4-r16-merger":   round((4 * 2 * 16 * 4096 * 28 * 4 + 4096*128 + 128*4 + 1280*4096 + 4096) / 1e6, 1),
}


def _match_catalogue(name: str):
    """Return the first catalogue entry whose substring appears in name."""
    for entry in _RUN_CATALOGUE:
        if entry[0] in name:
            return entry
    return None


def plot_architecture_comparison(records: dict, plot_dir: Path):
    """
    Two-panel figure summarising the priority ablation set:
      Left:  BERTScore-F1 grouped bar chart (baseline vs LoRA vs MoLoRA)
      Right: BERTScore-F1 vs trainable params scatter with connecting lines
             per architecture family.

    Handles missing runs gracefully — plots whatever data is available.
    """
    # Collect data
    rows = []
    for name, data in records.items():
        score = data["scores"].get("BERTScore-F1")
        if score is None:
            continue
        entry = _match_catalogue(name)
        if entry is None:
            continue
        substr, label, group, marker, cidx = entry
        params = _PARAMS_M.get(substr, None)
        rows.append({
            "name":    name,
            "label":   label,
            "group":   group,
            "marker":  marker,
            "colour":  _colour(cidx),
            "score":   score,
            "params":  params,
        })

    if not rows:
        print("No matching runs for architecture comparison — skipping")
        return

    # Sort by group then params for consistent ordering
    group_order = {"baseline": 0, "lora": 1, "molora": 2}
    rows.sort(key=lambda r: (group_order.get(r["group"], 9), r["params"] or 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: grouped bar chart ────────────────────────────────────────────
    bar_colours = [r["colour"] for r in rows]
    bar_labels  = [r["label"]  for r in rows]
    bar_scores  = [r["score"]  for r in rows]

    bars = ax1.bar(range(len(rows)), bar_scores, color=bar_colours,
                   edgecolor="white", linewidth=0.6, width=0.6)
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels(bar_labels, fontsize=8, rotation=25, ha="right")
    ax1.set_ylabel("BERTScore-F1")
    ax1.set_title("BERTScore-F1 by Configuration", fontweight="bold")
    ax1.set_ylim(0, max(bar_scores) * 1.18)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, bar_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(bar_scores) * 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Group separators
    group_counts = {}
    for r in rows:
        group_counts[r["group"]] = group_counts.get(r["group"], 0) + 1
    x = -0.5
    for g in ["baseline", "lora", "molora"]:
        c = group_counts.get(g, 0)
        if c:
            ax1.axvspan(x, x + c, alpha=0.04,
                        color={"baseline": "grey", "lora": "steelblue",
                               "molora": "darkorange"}[g])
            ax1.text(x + c / 2, max(bar_scores) * 1.10,
                     g.upper(), ha="center", fontsize=7.5,
                     color={"baseline": "grey", "lora": "steelblue",
                            "molora": "darkorange"}[g],
                     fontweight="bold")
            x += c

    # ── Right: params vs score scatter ────────────────────────────────────
    # Draw connecting lines per architecture family
    for group, lc in [("lora", "steelblue"), ("molora", "darkorange")]:
        grp_rows = [r for r in rows if r["group"] == group and r["params"] is not None]
        if len(grp_rows) >= 2:
            grp_rows.sort(key=lambda r: r["params"])
            ax2.plot([r["params"] for r in grp_rows],
                     [r["score"]  for r in grp_rows],
                     color=lc, linewidth=1.2, alpha=0.5, zorder=1)

    for r in rows:
        if r["params"] is None:
            continue
        ax2.scatter(r["params"], r["score"],
                    s=140, color=r["colour"], marker=r["marker"],
                    zorder=3, edgecolors="white", linewidths=0.6)
        ax2.annotate(r["label"], (r["params"], r["score"]),
                     textcoords="offset points", xytext=(6, 3), fontsize=7.5)

    ax2.set_xlabel("Trainable parameters (M)")
    ax2.set_ylabel("BERTScore-F1")
    ax2.set_title("Accuracy vs. Parameter Count\n"
                  "(○ MoLoRA  □ LoRA  ◇ zero-shot)", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("SPIQA+ Study Coach: Architecture Ablation Summary\n"
                 "(eval on 108 test examples)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = plot_dir / "architecture_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")




# ─────────────────────────────────────────────
# Plot 8: per-figure-type metric breakdown
# ─────────────────────────────────────────────

def plot_fig_type_metrics(records: dict, plot_dir: Path):
    """
    Grouped bar chart of BERTScore-F1 per figure type, one group per run.
    Shows whether different architectures are stronger/weaker on specific
    figure types (plot vs schematic vs table).
    Skips runs that have no by_fig_type data.
    """
    # Collect all runs that have by_fig_type data
    runs_with_data = {
        name: data for name, data in records.items()
        if data.get("by_fig_type")
    }
    if not runs_with_data:
        print("No per-figure-type data found — skipping fig_type_metrics plot")
        return

    # Get union of all figure types across runs, sorted
    all_fig_types = sorted({
        ft for data in runs_with_data.values()
        for ft in data["by_fig_type"].keys()
    })
    if not all_fig_types:
        return

    metric = "BERTScore-F1"
    n_types = len(all_fig_types)
    n_runs  = len(runs_with_data)

    fig, ax = plt.subplots(figsize=(max(8, n_types * n_runs * 0.5 + 2), 5))

    bar_width = 0.8 / n_runs
    x = np.arange(n_types)

    for i, (name, data) in enumerate(runs_with_data.items()):
        vals = [
            data["by_fig_type"].get(ft, {}).get(metric)
            for ft in all_fig_types
        ]
        # Skip if no values for this metric
        if all(v is None for v in vals):
            continue
        vals_plot = [v if v is not None else 0 for v in vals]
        offset = (i - n_runs / 2 + 0.5) * bar_width
        entry  = _match_catalogue(name)
        colour = _colour(entry[4]) if entry else _colour(i)
        label  = entry[1].replace("\n", " ") if entry else _short_label(name)
        bars = ax.bar(x + offset, vals_plot, bar_width * 0.9,
                      label=label, color=colour,
                      edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(all_fig_types, fontsize=10)
    ax.set_ylabel(f"{metric}")
    ax.set_title(f"{metric} by Figure Type\n(n=108)",
                 fontweight="bold")
    ax.legend(fontsize=7, ncol=min(n_runs, 4), loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # Add global score reference lines per run would clutter — just grid
    y_vals = [
        data["by_fig_type"].get(ft, {}).get(metric, 0)
        for data in runs_with_data.values()
        for ft in all_fig_types
    ]
    if y_vals:
        ax.set_ylim(min(y_vals) * 0.99, max(y_vals) * 1.015)

    plt.tight_layout()
    out = plot_dir / "fig_type_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

# ─────────────────────────────────────────────
# Plot 7: merger unfreezing comparison (Stage 2)
# ─────────────────────────────────────────────

def plot_merger_comparison(records: dict, plot_dir: Path):
    """
    2×2 comparison: frozen vs unfrozen merger, for LoRA r=64 and MoLoRA K=4 r=16.
    Left panel: grouped bar chart of BERTScore-F1.
    Right panel (MoLoRA only): router entropy comparison to show whether
    merger co-adaptation improves routing discriminability.
    """
    # Collect the four data points
    runs = {
        "lora-frozen":      ("baseline-lora-r64",       "LoRA r=64\nfrozen",      "steelblue",   "//"),
        "lora-merger":      ("lora-r64-merger",          "LoRA r=64\n+merger",     "steelblue",   ""),
        "molora-frozen":    ("molora-K4-r16",            "MoLoRA K=4 r=16\nfrozen","darkorange",  "//"),
        "molora-merger":    ("molora-K4-r16-merger",     "MoLoRA K=4 r=16\n+merger","darkorange", ""),
    }

    # Match records to run keys
    found = {}
    for key, (substr, label, colour, hatch) in runs.items():
        for name, data in records.items():
            if substr in name:
                found[key] = {
                    "label":   label,
                    "colour":  colour,
                    "hatch":   hatch,
                    "score":   data["scores"].get("BERTScore-F1"),
                    "entropy": data.get("routing", {}).get("router_entropy"),
                    "rouge":   data["scores"].get("ROUGE-L"),
                    "meteor":  data["scores"].get("METEOR"),
                }
                break

    if len(found) < 2:
        print("Not enough merger ablation runs for comparison plot — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: BERTScore-F1 grouped bars ───────────────────────────────────
    ax = axes[0]
    order = ["lora-frozen", "lora-merger", "molora-frozen", "molora-merger"]
    present = [k for k in order if k in found]
    xs      = range(len(present))
    bars = ax.bar(
        xs,
        [found[k]["score"] or 0 for k in present],
        color  = [found[k]["colour"] for k in present],
        hatch  = [found[k]["hatch"]  for k in present],
        edgecolor="white", linewidth=0.6, width=0.55,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([found[k]["label"] for k in present], fontsize=8)
    ax.set_ylabel("BERTScore-F1")
    ax.set_title("Effect of Merger Unfreezing\non Task Metrics", fontweight="bold")
    ax.set_ylim(min(v["score"] or 0 for v in found.values()) * 0.995,
                max(v["score"] or 0 for v in found.values()) * 1.010)
    ax.grid(axis="y", alpha=0.3)
    for bar, k in zip(bars, present):
        val = found[k]["score"]
        if val:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.0003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Draw delta arrows between frozen and merger pairs
    for frozen_k, merger_k, x_frozen, x_merger in [
        ("lora-frozen", "lora-merger", 0, 1),
        ("molora-frozen", "molora-merger", 2, 3),
    ]:
        if frozen_k in found and merger_k in found:
            v0 = found[frozen_k]["score"] or 0
            v1 = found[merger_k]["score"] or 0
            delta = v1 - v0
            mid_x = (x_frozen + x_merger) / 2
            y_top = max(v0, v1) + 0.001
            colour = "green" if delta >= 0 else "red"
            ax.annotate(
                f"Δ{delta:+.4f}",
                xy=(mid_x, y_top), fontsize=7.5,
                ha="center", color=colour, fontweight="bold"
            )

    # ── Right: router entropy comparison (MoLoRA only) ────────────────────
    ax2 = axes[1]
    molora_keys = ["molora-frozen", "molora-merger"]
    present_m   = [k for k in molora_keys if k in found and found[k]["entropy"] is not None]

    if present_m:
        xs2    = range(len(present_m))
        vals   = [found[k]["entropy"] for k in present_m]
        labels = [found[k]["label"]   for k in present_m]
        bars2  = ax2.bar(xs2, vals,
                         color=["darkorange", "chocolate"][:len(present_m)],
                         edgecolor="white", linewidth=0.6, width=0.4)
        ax2.axhline(np.log(4), color="steelblue", linestyle="--", alpha=0.5,
                    label=f"Max entropy ({np.log(4):.3f} nats)")
        ax2.set_xticks(xs2)
        ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_ylabel("Router Entropy (nats)")
        ax2.set_title("Effect of Merger Unfreezing\non Routing Entropy (MoLoRA)", fontweight="bold")
        ax2.set_ylim(0, np.log(4) * 1.25)
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars2, vals):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        if len(present_m) == 2:
            delta_e = vals[1] - vals[0]
            ax2.annotate(
                f"Δ{delta_e:+.3f} nats",
                xy=(0.5, max(vals) + 0.04), xycoords=("data", "data"),
                fontsize=8, ha="center",
                color="green" if delta_e >= 0 else "red", fontweight="bold"
            )
    else:
        ax2.text(0.5, 0.5, "MoLoRA merger run\nnot yet available",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)
        ax2.set_title("Routing Entropy (MoLoRA)", fontweight="bold")

    fig.suptitle("Stage 2: Merger Co-adaptation Ablation\n(hatched = merger frozen, solid = merger unfrozen)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = plot_dir / "merger_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default="./eval_results")
    p.add_argument("--plot_dir", default="./plots")
    return p.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    eval_dir = Path(args.eval_dir)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(eval_dir)
    if not records:
        print(f"No result JSONs found in {eval_dir}. Run evaluate_molora.py first.")
    else:
        print(f"Loaded {len(records)} result files:")
        for name, data in records.items():
            score  = data["scores"].get("BERTScore-F1")
            kind   = "MoLoRA" if data["is_molora"] else "baseline"
            s_str  = f"BERTScore-F1={score:.4f}" if score else "no score"
            print(f"  {name:45s} [{kind}]  {s_str}")
        print()
        plot_metrics_comparison(records, plot_dir)
        plot_routing_entropy(records, plot_dir)
        plot_expert_load_heatmap(records, plot_dir)
        plot_pareto(records, plot_dir)
        plot_temperature_ablation(records, plot_dir)
        plot_architecture_comparison(records, plot_dir)
        plot_merger_comparison(records, plot_dir)
        plot_fig_type_metrics(records, plot_dir)
        print(f"\nAll plots saved to {plot_dir}/")
