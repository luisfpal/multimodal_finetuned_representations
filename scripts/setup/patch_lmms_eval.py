#!/usr/bin/env python3
"""Fetch missing lmms-eval task template files from upstream.

lmms-eval ships task yamls that `include:` extension-less template files
(e.g. `_default_template_yaml`). setuptools globs only known extensions
when building the wheel, so these templates are silently omitted from the
pip package. At runtime, TaskManager walks every task dir and crashes on
the first unresolved include.

This script walks the installed lmms_eval/tasks/ tree, finds every
unresolved `include:` target (both quoted and unquoted yaml forms), and
downloads each missing file from upstream main. Run after every
`uv sync` until upstream fixes the wheel manifest.

Usage: python scripts/setup/patch_lmms_eval.py
"""

import os
import re
import sys
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/main/lmms_eval/tasks/"
INCLUDE_RE = re.compile(r'["\']?include["\']?\s*:\s*["\']?([^\s#\n"\']+)["\']?')
MAX_PASSES = 5


def locate_tasks_dir() -> str:
    try:
        import lmms_eval
    except ImportError:
        sys.exit("lmms_eval is not installed in the current environment")
    return os.path.join(os.path.dirname(lmms_eval.__file__), "tasks")


def scan_missing(tasks_dir: str) -> set[str]:
    missing: set[str] = set()
    for root, _, files in os.walk(tasks_dir):
        for fname in files:
            if not (fname.endswith((".yaml", ".yml")) or "yaml" in fname.lower()):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath) as fh:
                    content = fh.read()
            except OSError:
                continue
            for m in INCLUDE_RE.finditer(content):
                inc = m.group(1).strip()
                target = os.path.normpath(os.path.join(root, inc))
                if not os.path.exists(target):
                    missing.add(target)
    return missing


def fetch(local_path: str, tasks_dir: str) -> bool:
    rel = os.path.relpath(local_path, tasks_dir)
    url = BASE_URL + rel.replace(os.sep, "/")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = r.read()
    except Exception as e:
        print(f"  FAIL {rel}: {e}", file=sys.stderr)
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as fh:
        fh.write(data)
    return True


def main() -> int:
    tasks_dir = locate_tasks_dir()
    print(f"patching {tasks_dir}")

    total_fetched = 0
    total_failed = 0
    for pass_num in range(MAX_PASSES):
        missing = scan_missing(tasks_dir)
        if not missing:
            print(f"pass {pass_num}: nothing missing")
            break
        print(f"pass {pass_num}: {len(missing)} missing")
        pass_failed = 0
        for target in sorted(missing):
            if fetch(target, tasks_dir):
                total_fetched += 1
            else:
                pass_failed += 1
        if pass_failed == len(missing):
            total_failed = pass_failed
            break
    else:
        print(f"did not converge after {MAX_PASSES} passes", file=sys.stderr)
        return 1

    print(f"fetched: {total_fetched}, failed: {total_failed}")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
