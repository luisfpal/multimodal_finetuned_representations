#!/bin/bash
# =============================================================================
# Download SugarCrepe benchmark data.
#
# Sets up: ~/scratch/datasets/sugar-crepe/
#   data/   — 7 JSON annotation files from github.com/RAIVNLab/sugar-crepe
#   val2017/ — COCO 2017 validation images (~1 GB, 5000 images)
#
# Usage: bash shell/data/download_sugarcrepe.sh
# =============================================================================

# Hardcoded path: edit this path for your setup.
DEST="${SUGARCREPE_DATA_ROOT:-$HOME/scratch/datasets/sugar-crepe}"
DATA_DIR="$DEST/data"
IMG_DIR="$DEST/val2017"

echo "================================================================"
echo "  SugarCrepe Data Download"
echo "  Destination: $DEST"
echo "================================================================"
echo ""

mkdir -p "$DATA_DIR" "$IMG_DIR"

# ─── JSON annotation files ───────────────────────────────────────────────────
BASE_URL="https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data"
CATEGORIES=(add_obj add_att replace_obj replace_att replace_rel swap_obj swap_att)

echo "Downloading SugarCrepe JSON annotations..."
for cat in "${CATEGORIES[@]}"; do
    out="$DATA_DIR/$cat.json"
    if [ -f "$out" ]; then
        echo "  Already exists: $out"
    else
        echo "  Downloading $cat.json..."
        wget -q -O "$out" "$BASE_URL/$cat.json" || { echo "  ERROR: failed to download $cat.json"; exit 1; }
        echo "  Done: $out"
    fi
done
echo ""

# ─── COCO val2017 images ──────────────────────────────────────────────────────
if [ "$(ls -A "$IMG_DIR" 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "COCO val2017 images already present ($IMG_DIR) — skipping download."
else
    echo "Downloading COCO val2017 images (~1 GB)..."
    COCO_ZIP="/tmp/val2017_sugarcrepe_$$.zip"
    wget -O "$COCO_ZIP" "http://images.cocodataset.org/zips/val2017.zip" \
        || { echo "ERROR: failed to download COCO val2017"; rm -f "$COCO_ZIP"; exit 1; }
    echo "Extracting..."
    unzip -q "$COCO_ZIP" -d "$DEST/"
    rm -f "$COCO_ZIP"
    echo "Done: $IMG_DIR"
fi

echo ""
echo "================================================================"
echo "  SugarCrepe data ready at: $DEST"
printf "  JSON files: %d / %d\n" "$(ls "$DATA_DIR"/*.json 2>/dev/null | wc -l)" "${#CATEGORIES[@]}"
echo "  Images:     $(ls "$IMG_DIR" 2>/dev/null | wc -l)"
echo "================================================================"
