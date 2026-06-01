"""
canonicalize_human_charts.py
----------------------------
Remaps non-canonical coded values to codebook-valid categories across all
human-coded and compare chart JSONs. Re-run after every new data batch.

Covers:
  data/equity.json            — LLM vertical bar (x=labels, y=counts)
  human/equity.json           — Human vertical bar (x=labels, y=counts)
  compare/equity.json         — Grouped horizontal bar (x=%, y=labels, multi-trace)
  data/methodology.json       — LLM horizontal bar, multi-value ';'-separated y labels
  human/methodology.json      — Human horizontal bar (x=counts, y=labels)
  compare/methodology.json    — Grouped horizontal bar (x=%, y=labels, multi-trace)

Usage:
    python3 scripts/canonicalize_human_charts.py
"""

from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

FRONTEND_MAP = Path(__file__).resolve().parent.parent / "frontend" / "public" / "map"
DATA         = FRONTEND_MAP / "data"

# ── Canonical sets & mappings ─────────────────────────────────────────────────
# Source: CODEBOOK_FT.md, field: marginalized_subpopulations
EQUITY_ORDER: list[str] = [
    "women", "youth", "indigenous_peoples", "ethnic_minorities",
    "people_with_disabilities", "landless", "migrant_seasonal_workers",
    "other", "none_reported",
]
EQUITY_MAP: dict[str, str] = {
    # near-canonical spelling variants
    "indigenous_people":          "indigenous_peoples",
    "elders":                     "other",   # 'elderly' not in codebook → other
    "elderly":                    "other",
    "women farmers":              "women",
    "maasai pastoralists":        "ethnic_minorities",
    # not equity dimensions at all
    "small_holder_farmers":       "other",
    "small_holder_farmer":        "other",
    "smallholder rural farmers":  "other",
    "men":                        "other",
    "male_decision_making":       "other",
    "medium":                     "other",
    "poor":                       "other",
    "rich":                       "other",
    "rural":                      "other",
    "adults":                     "other",
    "poor rural farmers":         "other",
    "in doubt":                   "other",
    "others":                     "other",
}

# Source: CODEBOOK_FT.md, field: methodological_approach
METHOD_ORDER: list[str] = [
    "quantitative", "qualitative", "participatory",
    "modeling_with_empirical_validation", "experimental",
]
METHOD_MAP: dict[str, str] = {
    # free-text / misspelling variants
    "mixed_methods":                        "qualitative",
    "mixed methods":                        "qualitative",
    "mixed_method":                         "qualitative",
    "interviews":                           "qualitative",
    "semi-structured interviews":           "qualitative",
    "constant_comparative_methods":         "qualitative",
    "modelling with emperical validation":  "modeling_with_empirical_validation",
    "emperical_validation_modeling":        "modeling_with_empirical_validation",
    "sureys":                               "quantitative",   # typo: surveys
    "experimentation":                      "experimental",
    "participatory_methods":                "participatory",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _remap(raw: str, vmap: dict[str, str]) -> str:
    return vmap.get(raw.strip().lower(), raw.strip().lower())

def _split_method(raw: str) -> list[str]:
    """Expand ';'-separated multi-value methodology strings into canonical list."""
    out: list[str] = []
    for part in raw.split(";"):
        c = METHOD_MAP.get(part.strip().lower(), part.strip().lower())
        if c in METHOD_ORDER and c not in out:
            out.append(c)
    return out


def _fix_vbar(path: Path, vmap: dict[str, str], order: list[str]) -> None:
    """Vertical bar: x = category labels (str), y = counts (int)."""
    fig = json.loads(path.read_text())
    t = fig["data"][0]
    counts: dict[str, int] = defaultdict(int)
    for x, y in zip(t.get("x", []), t.get("y", [])):
        c = _remap(str(x), vmap)
        if c in order:
            counts[c] += int(y)
    ordered = [c for c in order if counts.get(c, 0) > 0]
    t["x"] = ordered
    t["y"] = [counts[c] for c in ordered]
    path.write_text(json.dumps(fig))
    print(f"  ✓ {path.parent.name}/{path.name}: {dict(zip(ordered, [counts[c] for c in ordered]))}")


def _fix_hbar(path: Path, vmap: dict[str, str], order: list[str]) -> None:
    """Horizontal bar: x = counts (int/float), y = category labels (str)."""
    fig = json.loads(path.read_text())
    t = fig["data"][0]
    counts: dict[str, float] = defaultdict(float)
    for y, x in zip(t.get("y", []), t.get("x", [])):
        c = _remap(str(y), vmap)
        if c in order:
            counts[c] += float(x)
    ordered = [c for c in order if counts.get(c, 0) > 0]
    t["y"] = ordered
    t["x"] = [round(counts[c]) for c in ordered]
    path.write_text(json.dumps(fig))
    print(f"  ✓ {path.parent.name}/{path.name}: {dict(zip(ordered, [round(counts[c]) for c in ordered]))}")


def _fix_hbar_multival(path: Path, order: list[str]) -> None:
    """LLM methodology: y labels are ';'-separated multi-value strings — split into canonical."""
    fig = json.loads(path.read_text())
    t = fig["data"][0]
    counts: dict[str, int] = defaultdict(int)
    for y, x in zip(t.get("y", []), t.get("x", [])):
        for c in _split_method(str(y)):
            counts[c] += int(x)
    ordered = [c for c in order if counts.get(c, 0) > 0]
    t["y"] = ordered
    t["x"] = [counts[c] for c in ordered]
    path.write_text(json.dumps(fig))
    print(f"  ✓ {path.parent.name}/{path.name}: {dict(zip(ordered, [counts[c] for c in ordered]))}")


def _fix_compare(path: Path, vmap: dict[str, str], order: list[str]) -> None:
    """Compare: multiple traces, x = % or counts (float), y = category labels (str)."""
    fig = json.loads(path.read_text())
    for t in fig["data"]:
        counts: dict[str, float] = defaultdict(float)
        for y, x in zip(t.get("y", []), t.get("x", [])):
            c = _remap(str(y), vmap)
            if c in order:
                counts[c] += float(x)
        ordered = [c for c in order if counts.get(c, 0) > 0]
        t["y"] = ordered
        t["x"] = [round(counts[c], 1) for c in ordered]
    path.write_text(json.dumps(fig))
    print(f"  ✓ {path.parent.name}/{path.name}: {ordered}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("canonicalize_human_charts.py\n")

    print("── Equity / marginalized_subpopulations ──")
    _fix_vbar(DATA / "equity.json",           EQUITY_MAP, EQUITY_ORDER)
    _fix_vbar(DATA / "human" / "equity.json", EQUITY_MAP, EQUITY_ORDER)
    _fix_compare(DATA / "compare" / "equity.json", EQUITY_MAP, EQUITY_ORDER)

    print("\n── Methodological approach ──")
    _fix_hbar_multival(DATA / "methodology.json",           METHOD_ORDER)
    _fix_hbar(DATA / "human" / "methodology.json",          METHOD_MAP, METHOD_ORDER)
    _fix_compare(DATA / "compare" / "methodology.json",     METHOD_MAP, METHOD_ORDER)

    print("\nDone.")


if __name__ == "__main__":
    main()
