"""
Analyse one or more results JSON files produced by the evaluation pipeline
and print a side-by-side comparison table across ablations.

Each entry in predictions[] has:
  ground_truth : "verdict = <v>\nerror_category = <e>\nfeedback = <f>"
  prediction   : same format

Metric abbreviations:
  PC Recall – Partially-Correct Recall
  FSR       – Feedback Suppression Rate
  VGS       – Visual Grounding Score

Usage:
    python analyse_r3.py <results1.json> [results2.json ...]
"""

import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_output(text: str) -> dict:
    """Extract verdict, error_category, and feedback from structured text."""
    result = {"verdict": None, "error_category": None, "feedback": None}
    if not text:
        return result

    verdict_m  = re.search(r"verdict\s*=\s*(.+)",        text, re.IGNORECASE)
    error_m    = re.search(r"error_category\s*=\s*(.+)", text, re.IGNORECASE)
    feedback_m = re.search(r"feedback\s*=\s*([\s\S]+)",  text, re.IGNORECASE)

    if verdict_m:
        result["verdict"]        = verdict_m.group(1).strip().lower()
    if error_m:
        result["error_category"] = error_m.group(1).strip().lower()
    if feedback_m:
        result["feedback"]       = feedback_m.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# VGS – Visual Grounding Score
# Checks whether the predicted feedback cites something visible in the figure.
# ---------------------------------------------------------------------------

_VGS_PATTERNS = re.compile(
    r"\b("
    r"figure|fig\.?|plot|table|graph|diagram|schematic|chart|image|illustration"
    r"|shown|shows|illustrate[sd]?|depicted|displayed|visible|indicate[sd]?"
    r"|x-axis|y-axis|axis|axes"
    r"|bar|bars|curve|curves|line|lines|point|points"
    r"|column|columns|row|rows|cell|cells"
    r"|legend|label|caption"
    r"|solid|dashed|dotted|red|blue|green|black|orange|purple"
    r"|left|right|top|bottom|upper|lower"
    r"|value|values|number|percentage|percent|%"
    r")\b",
    re.IGNORECASE,
)

def cites_figure(feedback: str) -> bool:
    """Return True if feedback contains language grounded in the figure."""
    if not feedback or feedback.strip().upper() == "N/A":
        return False
    return bool(_VGS_PATTERNS.search(feedback))


# ---------------------------------------------------------------------------
# Per-file metric computation
# ---------------------------------------------------------------------------

def compute_metrics(path: str) -> dict:
    """
    Load a results JSON and return a dict of computed metrics plus metadata.
    """
    with open(path) as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    if not predictions:
        raise ValueError(f"No predictions found in {path}")

    total = len(predictions)

    # Verdict / error counters
    verdict_match      = 0   # gt verdict == pred verdict
    error_match        = 0   # gt error_category == pred error_category
    verdict_only_match = 0   # verdict matches but error_category differs
    both_match         = 0   # both verdict AND error_category match

    # PC Recall – Partially-Correct Recall
    pc_gt_total    = 0
    pc_gt_pred_pc  = 0

    # FSR – Feedback Suppression Rate
    # Predicted as correct though ground truth is incorrect or partially correct
    non_correct_gt   = 0
    fsr_pred_correct = 0

    # VGS – Visual Grounding Score
    # Fraction of non-correct (incorrect or partially correct) feedback that cites figure content
    vgs_gt_grounded   = 0
    vgs_pred_grounded = 0

    parse_errors = 0

    for ex in predictions:
        gt   = parse_output(ex.get("ground_truth", ""))
        pred = parse_output(ex.get("prediction",   ""))

        if gt["verdict"] is None or pred["verdict"] is None:
            parse_errors += 1
            continue

        gt_v  = gt["verdict"]
        gt_e  = gt["error_category"] or ""
        gt_fb = gt["feedback"] or ""
        pr_v  = pred["verdict"]
        pr_e  = pred["error_category"] or ""
        pr_fb = pred["feedback"] or ""

        v_ok = (gt_v == pr_v)
        e_ok = (gt_e == pr_e)

        if v_ok:
            verdict_match += 1
        if e_ok:
            error_match += 1
        if v_ok and not e_ok:
            verdict_only_match += 1
        if v_ok and e_ok:
            both_match += 1

        # PC Recall – Partially-Correct Recall
        if gt_v == "partially correct":
            pc_gt_total += 1
            if pr_v == "partially correct":
                pc_gt_pred_pc += 1

        # FSR – Feedback Suppression Rate
        if gt_v in ("incorrect", "partially correct"):
            non_correct_gt += 1
            if pr_v == "correct":
                fsr_pred_correct += 1

        # VGS – Visual Grounding Score (only for non-correct ground truth examples)
        if gt_v in ("incorrect", "partially correct"):
            if cites_figure(gt_fb):
                vgs_gt_grounded += 1
            if cites_figure(pr_fb):
                vgs_pred_grounded += 1

    def pct(num, den):
        return 100.0 * num / den if den else float("nan")

    return {
        "name":         data.get("ablation", Path(path).stem),
        "model_id":     data.get("model_id", "unknown"),
        "total":        total,
        "parse_errors": parse_errors,
        # Metrics
        "verdict_acc":      pct(verdict_match,      total),
        "error_acc":        pct(error_match,         total),
        "verdict_only_acc": pct(verdict_only_match,  total),
        "class_acc":        pct(both_match,          total),
        "pc_recall":        pct(pc_gt_pred_pc,       pc_gt_total),      # Partially-Correct Recall
        "fsr":              pct(fsr_pred_correct,    non_correct_gt),   # Feedback Suppression Rate
        "vgs_gt":           pct(vgs_gt_grounded,   non_correct_gt), # Visual Grounding Score (ground truth)
        "vgs_pred":         pct(vgs_pred_grounded, non_correct_gt), # Visual Grounding Score (prediction)
        # Supporting counts for context
        "pc_gt_total":    pc_gt_total,
        "non_correct_gt": non_correct_gt,
    }


# ---------------------------------------------------------------------------
# Display comparison table
# ---------------------------------------------------------------------------

METRICS_CORE = [
    ("verdict_acc",      "Verdict Accuracy (%)"),
    ("pc_recall",        "PC Recall (%) [Partially-Correct Recall]"),
    ("fsr",              "FSR (%) [Feedback Suppression Rate]"),
    ("vgs_gt",           "VGS GT (%) [Visual Grounding Score]"),
    ("vgs_pred",         "VGS Pred (%) [Visual Grounding Score]"),
]

METRICS_EXTRA = [
    ("error_acc",        "Error Accuracy (%)"),
    ("verdict_only_acc", "Verdict-Only Accuracy (%)"),
    ("class_acc",        "Classification Accuracy (%)"),
]

def active_metrics(extra: bool) -> list:
    return METRICS_CORE + METRICS_EXTRA if extra else METRICS_CORE

def print_comparison(results: list[dict], extra: bool = False) -> None:
    col_w   = max(14, max(len(r["name"]) for r in results) + 2)  # fit longest name + padding
    label_w = 44          # width of metric label column

    # Header: ablation names
    header = f"{'Metric':<{label_w}}" + "".join(
        f"{r['name']:>{col_w}}" for r in results
    )
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("  Ablation Comparison")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for key, label in active_metrics(extra):
        row = f"{label:<{label_w}}"
        for r in results:
            val = r[key]
            row += f"{val:>{col_w}.1f}"
        print(row)

    print(sep)

    # Footer: metadata rows
    for field, label in [
        ("total",           "Total examples"),
        ("parse_errors",    "Parse errors"),
        ("pc_gt_total",     "  GT partially-correct"),
        ("non_correct_gt", "  GT non-correct"),
    ]:
        row = f"{label:<{label_w}}"
        for r in results:
            row += f"{r[field]:>{col_w}}"
        print(row)

    print()

    # Model IDs (can be long – print separately)
    print("Models:")
    for r in results:
        print(f"  {r['name']:<30} {r['model_id']}")
    print()


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def escape_latex(s: str) -> str:
    """Escape special LaTeX characters in a string."""
    return (s.replace("&", r"\&")
             .replace("%", r"\%")
             .replace("_", r"\_")
             .replace("#", r"\#")
             .replace("{", r"\{")
             .replace("}", r"\}"))

def gen_latex(results: list[dict], out_path: str, extra: bool = False) -> None:
    n_cols = len(results)
    col_spec = "l" + "r" * n_cols  # left-aligned metric, right-aligned data columns

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # Header row: ablation names
    header_cells = ["Metric"] + [escape_latex(r["name"]) for r in results]
    lines.append("    " + " & ".join(header_cells) + r" \\")
    lines.append(r"    \midrule")

    # Metric rows
    for key, label in active_metrics(extra):
        # Strip the bracketed expansion from label for a cleaner table
        short_label = escape_latex(label.split(" [")[0])
        cells = [short_label] + [f"{r[key]:.1f}" for r in results]
        lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \caption{Ablation comparison}")
    lines.append(r"  \label{tab:ablation}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"LaTeX table written to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import glob as _glob

    parser = argparse.ArgumentParser(
        description="Analyse and compare evaluation result JSON files."
    )
    parser.add_argument(
        "files", nargs="+",
        help="Result JSON files or glob patterns (e.g. 'results/*.json')"
    )
    parser.add_argument(
        "--gen_latex", metavar="OUTPUT.tex", default=None,
        help="Also write a LaTeX comparison table to this .tex file"
    )
    parser.add_argument(
        "--extra", action="store_true",
        help="Include Error Accuracy, Verdict-Only Accuracy, and Classification Accuracy"
    )
    args = parser.parse_args()

    # Expand glob patterns and deduplicate while preserving order
    paths = []
    seen  = set()
    for arg in args.files:
        expanded = sorted(_glob.glob(arg, recursive=True)) or [arg]
        for p in expanded:
            if p not in seen:
                seen.add(p)
                paths.append(p)

    all_results = []
    for path in paths:
        try:
            all_results.append(compute_metrics(path))
        except Exception as e:
            print(f"[Error] {path}: {e}", file=sys.stderr)

    if not all_results:
        sys.exit(1)

    print_comparison(all_results, extra=args.extra)

    if args.gen_latex:
        gen_latex(all_results, args.gen_latex, extra=args.extra)
