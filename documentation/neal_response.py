"""
neal_response.py

Copies RESPONSE_TO_NEAL.md into the neal_response output folder and
generates supporting figures (kappa convergence chart).

The markdown file is the canonical response document.

Run from anywhere:
    conda run -n ilri01 python documentation/neal_response.py

Outputs:
    documentation/neal_response/
        RESPONSE_TO_NEAL.md
        figures/
            kappa_convergence.png
"""

import io
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE    = Path(__file__).resolve().parent.parent   # project root
OUTPUTS = HERE / "scripts" / "outputs"
OUT_DIR = Path(__file__).resolve().parent / "neal_response"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MD_SRC  = HERE / "RESPONSE_TO_NEAL.md"

# ---------------------------------------------------------------------------
# Load calibration data
# ---------------------------------------------------------------------------
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

c_r1  = load_json(OUTPUTS / "step11" / "EPPI_Review_-_R1_summary.json")
c_r1b = load_json(OUTPUTS / "step11" / "EPPI_Review_-_R1b_summary.json")
c_r2  = load_json(OUTPUTS / "step11" / "EPPI_Review_-_R2_summary.json")
c_r2a = load_json(OUTPUTS / "step11" / "EPPI_Review_-_R2a_summary.json")
c_r3a = load_json(OUTPUTS / "step11" / "EPPI_Review_-_R3a_summary.json")

pk1  = c_r1.get("pairwise_kappa",  {})
pk1b = c_r1b.get("pairwise_kappa", {})
pk2  = c_r2.get("pairwise_kappa",  {})
pk2a = c_r2a.get("pairwise_kappa", {})
pk3  = c_r3a.get("pairwise_kappa", {})

# ---------------------------------------------------------------------------
# Kappa convergence figure
# ---------------------------------------------------------------------------
def make_kappa_figure() -> Path:
    rounds = ["R1\n(n=205)", "R1b\n(n=205)", "R2\n(n=103)", "R2a\n(n=103)", "R3a\n(n=107)"]
    llm_kappa = [
        pk1.get("LLM vs CJ Reconciled",     0.436),
        pk1b.get("LLMr1b vs CJ Reconciled", 0.645),
        pk2.get("Jennifer Cisse vs LLM",     0.717),
        pk2a.get("LLM_r2a vs CJ Reconciled", 0.770),
        float(np.mean([
            pk3.get("Jennifer Cisse vs LLM",  0.690),
            pk3.get("Caroline Staub vs LLM",  0.674),
        ])),
    ]
    human_kappa = [
        pk1.get("Caroline Staub vs Jennifer Cisse",  0.500),
        pk1b.get("Caroline Staub vs Jennifer Cisse", 0.500),
        pk2.get("Jennifer Cisse vs Caroline Staub",  0.765),
        pk2a.get("Caroline Staub vs Jennifer Cisse", 0.765),
        pk3.get("Jennifer Cisse vs Caroline Staub",  0.703),
    ]

    x = np.arange(len(rounds))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, llm_kappa,   "o-",  color="#2166ac", linewidth=2, markersize=7,
            label="LLM vs reconciled gold standard")
    ax.plot(x, human_kappa, "s--", color="#d73027", linewidth=2, markersize=7,
            label="Human inter-rater (Caroline vs Jennifer)")
    ax.axhspan(0.61, 0.80, color="#fee08b", alpha=0.25, label="Substantial (0.61–0.80)")
    ax.axhspan(0.80, 1.00, color="#a6d96a", alpha=0.25, label="Almost perfect (> 0.80)")
    ax.axhline(0.61, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(0.80, color="gray", linewidth=0.8, linestyle=":")
    for i, label in [(0, "Criteria\nv1"), (1, "Criteria\nv1b"), (3, "Criteria\nv2a")]:
        ax.annotate(label, xy=(i, llm_kappa[i]),
                    xytext=(i + 0.08, llm_kappa[i] - 0.07),
                    fontsize=7, color="#2166ac",
                    arrowprops=dict(arrowstyle="-", color="#2166ac", lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(rounds, fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Cohen's kappa", fontsize=10)
    ax.set_title("LLM calibration convergence across rounds", fontsize=11, pad=8)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    dest = FIG_DIR / "kappa_convergence.png"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(dest, format="PNG", dpi=(150, 150))
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Copy markdown
    if MD_SRC.exists():
        shutil.copy2(MD_SRC, OUT_DIR / "RESPONSE_TO_NEAL.md")
        print(f"Copied: {OUT_DIR / 'RESPONSE_TO_NEAL.md'}")
    else:
        print(f"Warning: {MD_SRC} not found")

    # Generate figure
    fig_path = make_kappa_figure()
    print(f"Saved:  {fig_path}")


if __name__ == "__main__":
    main()
