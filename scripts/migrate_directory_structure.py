#!/usr/bin/env python3
"""
migrate_directory_structure.py
──────────────────────────────
One-shot migration from the old directory layout to the new clean hierarchy.

OLD LAYOUT
──────────
Representations (on scratch):
  {model}/{dataset}_{caption_source}[__pool-{method}]/output_layer/{method}/representations.safetensors
  - Images have no caption_source suffix; pooling bleeds into folder name

Results:
  multimodality_comparison/{dataset}_{caption_source}_img_vs_txt[__pool-{method}][__{sem_tag}]/
    similarities/{model}_{pooling}_{metric}.safetensors

NEW LAYOUT
──────────
Representations (on scratch):
  {model}/{dataset}_{caption_source}_{txt_or_img}/output_layer/{pooling}/[sf-{tag}/]representations.safetensors
  - Image folders never have caption_source (they're caption-agnostic)
  - No __pool-X suffix anywhere; pooling is the leaf directory

Results:
  multimodality_comparison/{dataset}_{caption_source}/{pooling}/[sf-{tag}/]similarities/{model}_{metric}.safetensors
  - No _img_vs_txt suffix; no __pool-X suffix
  - Model filename no longer encodes pooling

USAGE
──────
# Preview what will happen (default: dry-run)
python scripts/migrate_directory_structure.py \\
    --representations-dir ~/scratch \\
    --results-dir ~/multimodal_finetuned_representations/results \\
    --dry-run

# Execute migration
python scripts/migrate_directory_structure.py \\
    --representations-dir ~/scratch \\
    --results-dir ~/multimodal_finetuned_representations/results \\
    --execute
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path


# ==============================================================================
# HELPERS
# ==============================================================================

def _move(src: Path, dst: Path, dry_run: bool) -> bool:
    """
    Move src to dst, creating parent dirs as needed.
    Returns True if the move was (or would be) performed.
    """
    if not src.exists():
        return False
    if dst.exists():
        print(f"  ⚠  SKIP (dst exists): {dst}")
        return False
    print(f"  {'[DRY]' if dry_run else '      '} {src}")
    print(f"         → {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return True


def _rmdir_if_empty(path: Path, dry_run: bool):
    """Remove path if it is an empty directory (recursively up to 3 levels)."""
    for _ in range(3):
        if path.exists() and path.is_dir():
            try:
                children = list(path.iterdir())
                if not children:
                    print(f"  {'[DRY]' if dry_run else '      '} rmdir {path}")
                    if not dry_run:
                        path.rmdir()
                    path = path.parent
                    continue
            except Exception:
                pass
        break


# ==============================================================================
# REPRESENTATIONS MIGRATION
# ==============================================================================

# Pattern for old repr folder names with __pool-{method} suffix
# Matches: {prefix}__pool-{method}
_REPR_POOL_SUFFIX_RE = re.compile(r"^(.+)__pool-(.+)$")

# Pattern for old repr folder names WITHOUT __pool suffix but WITH _img_vs_txt
# (these shouldn't appear in repr dirs, but guard defensively)

def migrate_representations(repr_base: Path, dry_run: bool):
    """
    Migrate representations directory.

    For each model dir, for each dataset folder that has __pool-{method} suffix:
      OLD: {model}/{dataset}__pool-{method}/output_layer/{method}/[representations.safetensors]
      NEW: {model}/{dataset}/output_layer/{method}/[representations.safetensors]

    Semantic filter subdir (sf-{tag}) is already correctly nested under {method}/ in
    the old code; no change needed there — only the __pool-X folder suffix is removed.
    """
    reprs_dir = repr_base / "representations"
    if not reprs_dir.exists():
        print(f"  ⚠  Representations dir not found: {reprs_dir}")
        return

    moves = 0
    model_dirs = [d for d in reprs_dir.iterdir() if d.is_dir()]

    print(f"\n{'─'*66}")
    print(f"  REPRESENTATIONS: {reprs_dir}")
    print(f"{'─'*66}")

    for model_dir in sorted(model_dirs):
        dataset_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        for dataset_dir in sorted(dataset_dirs):
            m = _REPR_POOL_SUFFIX_RE.match(dataset_dir.name)
            if not m:
                continue  # No __pool-X suffix → already in new format
            base_name = m.group(1)  # e.g. "cocoqa_restval_icqa_human_txt"
            method = m.group(2)     # e.g. "mean_context"

            # Old:  {model}/{dataset}__pool-{method}/output_layer/{method}/
            # New:  {model}/{base_name}/output_layer/{method}/
            old_method_dir = dataset_dir / "output_layer" / method
            new_method_dir = model_dir / base_name / "output_layer" / method

            if not old_method_dir.exists():
                print(f"  ⚠  Missing method dir: {old_method_dir}")
                continue

            # Move the pooling directory itself (and everything inside it)
            if _move(old_method_dir, new_method_dir, dry_run):
                moves += 1
                _rmdir_if_empty(dataset_dir / "output_layer", dry_run)
                _rmdir_if_empty(dataset_dir, dry_run)

    print(f"\n  Representations: {moves} move(s) {'planned' if dry_run else 'completed'}")


# ==============================================================================
# RESULTS MIGRATION
# ==============================================================================

# Old folder naming scheme for results:
#   {dataset}_{caption_source}_img_vs_txt[__pool-{method}][__{sem_tag}]
# We need to:
#   1. Strip _img_vs_txt
#   2. Extract pooling method (default "last" if no __pool-X)
#   3. Extract optional semantic tag
#   4. Rename files: {model}_{pooling}_{metric}.safetensors → {model}_{metric}.safetensors

_KNOWN_POOLING_METHODS = ["last", "mean_context"]
# Sem tag pattern — the text after the second __ (if present)
_RESULTS_FOLDER_RE = re.compile(
    r"^(?P<base>.+)_img_vs_txt(?:__pool-(?P<pooling>(?:(?!__).)+))?(?:__(?P<sem>.+))?$"
)


def _strip_pooling_from_filename(stem: str, pooling: str) -> str:
    """
    Remove the pooling component from a safetensors filename stem.

    Old: llava_7b_last_neighborhood_overlap
    New: llava_7b_neighborhood_overlap

    Old: InternVL2-8B_mean_context_linear_cka
    New: InternVL2-8B_linear_cka

    Strategy: remove the first occurrence of _{pooling}_ (surrounded by underscores).
    This is safe because pooling methods are "last" and "mean_context" which don't
    appear as part of model names or metric names.
    """
    token = f"_{pooling}_"
    idx = stem.find(token)
    if idx == -1:
        # Possibly at the end (shouldn't happen for safetensors with metric suffix)
        # Try exact suffix
        if stem.endswith(f"_{pooling}"):
            return stem[: -len(f"_{pooling}")]
        return stem  # couldn't find it; return unchanged (safe)
    return stem[:idx] + "_" + stem[idx + len(token):]


def migrate_results(results_dir: Path, dry_run: bool):
    """
    Migrate results/multimodality_comparison directory.

    For each old-style folder:
      OLD: {base}_img_vs_txt[__pool-{method}][__{sem}]/similarities/{model}_{pooling}_{metric}.safetensors
      NEW: {base}/{pooling}/[sf-{sem}/]similarities/{model}_{metric}.safetensors
    """
    mc_dir = results_dir / "multimodality_comparison"
    if not mc_dir.exists():
        print(f"  ⚠  Results dir not found: {mc_dir}")
        return

    moves = 0
    renamed = 0
    print(f"\n{'─'*66}")
    print(f"  RESULTS: {mc_dir}")
    print(f"{'─'*66}")

    for folder in sorted(mc_dir.iterdir()):
        if not folder.is_dir():
            continue
        m = _RESULTS_FOLDER_RE.match(folder.name)
        if not m:
            # Check if already in new format (no _img_vs_txt)
            # New format dirs are plain "{dataset}_{caption_source}" or "{pooling}"
            # We skip them silently
            if "_img_vs_txt" not in folder.name:
                continue
            print(f"  ⚠  Cannot parse folder name: {folder.name}")
            continue

        base = m.group("base")       # e.g. "cocoqa_restval_icqa_human"
        pooling = m.group("pooling") or "last"   # default to "last" when no __pool-X
        sem = m.group("sem") or ""   # e.g. "sf-jina_cpu_m1sd" or ""

        # Normalise: old sem tags didn't have "sf-" prefix in some versions
        # Current code produces sem_tag = "sf-jina_cpu_m1sd" already,
        # but older data may have it as "__sf-jina_cpu_m1sd"; handle either.
        sem_dir = ""
        if sem:
            sem_dir = sem if sem.startswith("sf-") else f"sf-{sem}"

        # Build new similarities dir
        new_sim_dir = mc_dir / base / pooling
        if sem_dir:
            new_sim_dir = new_sim_dir / sem_dir
        new_sim_dir = new_sim_dir / "similarities"

        old_sim_dir = folder / "similarities"
        if not old_sim_dir.exists():
            print(f"  ⚠  No similarities/ in: {folder.name}")
            continue

        safetensor_files = list(old_sim_dir.glob("*.safetensors"))
        if not safetensor_files:
            print(f"  ·  Empty similarities/: {folder.name}")
            continue

        print(f"\n  Folder: {folder.name}")
        print(f"  → {base}/{pooling}{'/' + sem_dir if sem_dir else ''}/similarities/")

        for old_path in sorted(safetensor_files):
            # Rename: strip pooling from filename
            new_stem = _strip_pooling_from_filename(old_path.stem, pooling)
            new_name = new_stem + old_path.suffix
            new_path = new_sim_dir / new_name

            if new_stem != old_path.stem:
                renamed += 1

            if _move(old_path, new_path, dry_run):
                moves += 1

        # Remove old similarities dir and folder if now empty
        _rmdir_if_empty(old_sim_dir, dry_run)
        _rmdir_if_empty(folder, dry_run)

    print(f"\n  Results: {moves} file move(s) {'planned' if dry_run else 'completed'}")
    print(f"           {renamed} filename(s) had pooling stripped")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Migrate representations and results to the new directory structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--representations-dir",
        type=str,
        required=True,
        help="Base scratch directory (contains representations/ subfolder)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Project results directory (contains multimodality_comparison/ subfolder)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview moves without touching disk (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration",
    )
    args = parser.parse_args()

    dry_run = not args.execute

    repr_base = Path(os.path.expanduser(args.representations_dir))
    results_dir = Path(os.path.expanduser(args.results_dir))

    print("=" * 66)
    mode_str = "DRY RUN (preview only)" if dry_run else "EXECUTE (moving files!)"
    print(f"  Directory Migration  —  {mode_str}")
    print("=" * 66)

    if not dry_run:
        print("\n  ⚠  This will MOVE files on disk. Make sure you have reviewed")
        print("     the dry-run output first!")
        resp = input("  Continue? [y/N] ").strip().lower()
        if resp != "y":
            print("  Aborted.")
            sys.exit(0)

    migrate_representations(repr_base, dry_run)
    migrate_results(results_dir, dry_run)

    print("\n" + "=" * 66)
    if dry_run:
        print("  DRY RUN complete. Run with --execute to apply changes.")
    else:
        print("  Migration complete.")
    print("=" * 66)


if __name__ == "__main__":
    main()
