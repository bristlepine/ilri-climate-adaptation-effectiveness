"""Generate PNG exports for all human-coded figures."""
import json
import plotly.graph_objects as go
from pathlib import Path

HUMAN  = Path(__file__).parent.parent / "scripts" / "outputs" / "step16" / "interactive" / "human"
STEP16 = Path(__file__).parent.parent / "scripts" / "outputs" / "step16"

needed = [
    ("geographic_map",   900, 480),
    ("geographic_bar",   900, 460),
    ("methodology",      900, 440),
    ("temporal_trends",  900, 440),
    ("equity",           900, 440),
    ("domain_heatmap",  1000, 500),
    ("domain_type",      900, 420),
    ("producer_type",    900, 420),
]

for name, w, h in needed:
    jp = HUMAN / f"{name}.json"
    pp = HUMAN / f"{name}.png"
    if not jp.exists():
        print(f"MISS: {name}")
        continue
    with open(jp) as f:
        fig = go.Figure(json.load(f))
    fig.write_image(str(pp), width=w, height=h, scale=2)
    print(f"OK: {pp.name}")

# Saturation
sat_j = Path(__file__).parent.parent / "scripts" / "outputs" / "step15c" / "saturation.json"
sat_p = STEP16 / "saturation.png"
if sat_j.exists() and not sat_p.exists():
    with open(sat_j) as f:
        fig = go.Figure(json.load(f))
    fig.write_image(str(sat_p), width=1100, height=620, scale=2)
    print("OK: saturation.png")
elif sat_p.exists():
    print("skip: saturation.png (exists)")
else:
    print("MISS: saturation.json")

print("Done.")
