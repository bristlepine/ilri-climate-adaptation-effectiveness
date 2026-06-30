[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17809739.svg)](https://doi.org/10.5281/zenodo.17809739)

# Measuring What Matters: Tracking Climate Adaptation for Smallholder Producers

ILRI / Bristlepine Resilience Consultants — 2025–2026

Systematic map and evidence synthesis on methods used to track climate adaptation processes and outcomes for smallholder producers in LMICs. Follows CEE/Campbell standards.

---

## Setup

```bash
git clone https://github.com/bristlepine/ilri-climate-adaptation-effectiveness.git
cd ilri-climate-adaptation-effectiveness
conda env create -f environment.yml
conda activate ilri01
```

Place the `.env` file (API keys) in the repo root. Place the `outputs/` folder at `scripts/outputs/`.

---

## Running the pipeline

Run individual steps:

```bash
conda run -n ilri01 python scripts/step12_screen_abstracts.py
```

Or use the orchestrator with flags set in `scripts/config.py`:

```bash
conda run -n ilri01 python scripts/run.py
```

To run specific steps only:

```bash
conda run -n ilri01 python scripts/run.py --step=15,16
```

All steps are resume-safe — interrupt and restart without reprocessing completed records.

See [METHODS.md](METHODS.md) for a full walkthrough of each pipeline step.

---

## Citation

Cissé, J. D., Staub, C. G., & Khan, Z. (2025). *Measuring what matters: tracking climate adaptation processes and outcomes for smallholder producers in the agriculture sector.* Zenodo. https://doi.org/10.5281/zenodo.17809739

---

## Links

| | |
|---|---|
| GitHub | https://github.com/bristlepine/ilri-climate-adaptation-effectiveness |
| Zenodo | https://doi.org/10.5281/zenodo.17809739 |
| Live site | https://bristlepine.com |
| Google Drive | https://drive.google.com/drive/folders/1f5y8kjVAcHXBm74AM2wOXsdxrCTnh-ll |
