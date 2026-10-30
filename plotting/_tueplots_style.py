"""Matplotlib rcParams for paper figures, built from the tueplots neurips2024 bundle."""

from pathlib import Path

USE_TUEPLOTS = True

OUTPUT_DIR = Path(__file__).resolve().parent / "tueplots"
OUTPUT_DIR.mkdir(exist_ok=True)

if USE_TUEPLOTS:
    from tueplots import bundles, fontsizes


def get_rc(ncols=2, nrows=1, **overrides):
    if not USE_TUEPLOTS:
        return dict(overrides)
    rc = bundles.neurips2024(nrows=nrows, ncols=ncols)
    rc.update(fontsizes.neurips2024())
    rc["text.usetex"] = False
    
    # Scale up all font sizes by 2 points for readability
    for key in list(rc.keys()):
        if "size" in key and isinstance(rc[key], (int, float)):
            rc[key] += 3
            
    rc.update(overrides)
    return rc


def out_path(name):
    return str(OUTPUT_DIR / name)
