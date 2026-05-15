import os
import json
import numpy as np
from pathlib import Path

EVALS_ROOT = Path('/u/dssc/lbasile/scratch/plm/evals')

MODELS = [
    ('llava_7b', 'LLaVA-7B'),
    ('llava_13b', 'LLaVA-13B'),
    ('LLaVA-OneVision-1.5-4B', 'OneVision-4B-RD'),
    ('LLaVA-OneVision-1.5-4B-Instruct', 'OneVision-4B'),
    ('LLaVA-OneVision-1.5-8B-Instruct', 'OneVision-8B'),
    ('InternVL2-8B', 'InternVL2-8B'),
    ('cambrian-8b', 'Cambrian-8B'),
    (None, 'Llama-90B'),
]

OPEN_ENDED_BENCHMARKS = [
    ('GQA',           'Exact Match (\\%)',     'gqa',                 'exact_match,none',         1,    100),
    ('VQAv2',         'Exact Match (\\%)',     'vqav2_val',           'exact_match,none',         1,    100),
    ('CountBench',    'Accuracy (\\%)',        'countbench',          'acc,none',                 1,    100),
    ('ChartQA',       'Relaxed Acc. (\\%)',    'chartqa',             'relaxed_overall,none',     1,    100),
]

MULTIPLE_CHOICE_BENCHMARKS = [
    ('ScienceQA',     'Exact Match (\\%)',     'scienceqa_img',       'exact_match,none',         1,    100),
    ('MMBench-EN',    'GPT Score',             'mmbench_en_dev',      'gpt_eval_score,none',      1,    1),
    ('MMBench-CN',    'GPT Score',             'mmbench_cn_dev',      'gpt_eval_score,none',      1,    1),
    ('MME',           'Normalized Score (\\%)', 'mme',                'mme_total',                2800, 100),
    ('POPE',          'F1 Score (\\%)',        'pope',                'pope_f1_score,none',       1,    100),
    ('SEED',          'Accuracy (\\%)',        'seedbench',           'seed_all,none',            1,    100),
    ('MMMU',          'Accuracy (\\%)',        'mmmu_val',            'mmmu_acc,none',            1,    100),
    ('MMMU-Pro',      'Accuracy (\\%)',        'mmmu_pro_standard',   'mmmu_acc,none',            1,    100),
    ('MMStar',        'Mean Acc. (\\%)',       'mmstar',              'average,none',             1,    100),
    ('NaturalBench',  'Group Acc. (\\%)',      'naturalbench',        'naturalbench_G_ACC,none',  1,    100),
    ('CVBench',       'Accuracy (\\%)',        'cv_bench',            'cv_bench_acc,none',        1,    100),
    ('RealWorldQA',   'Exact Match (\\%)',     'realworldqa',         'exact_match,flexible-extract', 1, 100),
]

COMP_BENCHMARKS = [
    ('ARO',           'Mean Acc. (\\%)',       '../compositionality/aro',          'accuracy',  1, 100),
    ('SugarCrepe',    'Mean Acc. (\\%)',       '../compositionality/sugarcrepe',   'mean',      1, 100),
]

CAP_BENCHMARKS = [
    ('COCO',     'CIDEr ($\\times 100$)', 'coco2017_cap_val', 'coco_CIDEr,none',    1, 100),
    ('Flickr30k','CIDEr ($\\times 100$)', 'flickr30k',        'flickr_CIDEr,none',  1, 100),
]


def get_score(model_dir: Path, bench_subdir: str, key: str) -> float:
    bench_path = model_dir / 'lmms_eval' / bench_subdir
    if not bench_path.exists():
        return None
    files = [f for f in os.listdir(bench_path) if f.endswith('.jsonl') and 'results' in f]
    if not files:
        return None

    if bench_subdir.endswith('aro'):
        vals = []
        for rf in sorted(files):
            with open(bench_path / rf) as f:
                lines = f.readlines()
            if not lines:
                continue
            res = json.loads(lines[-1])
            if key in res:
                vals.append(res[key])
        return float(np.mean(vals)) if vals else None

    if bench_subdir.endswith('sugarcrepe'):
        with open(bench_path / files[0]) as f:
            lines = f.readlines()
        if not lines:
            return None
        res = json.loads(lines[-1])
        return res.get(key)

    with open(bench_path / files[0]) as f:
        lines = f.readlines()
    if not lines:
        return None
    last = json.loads(lines[-1])

    if key == 'mme_total':
        p = last.get('mme_perception_score,none', 0)
        c = last.get('mme_cognition_score,none', 0)
        return p + c

    return last.get(key)


def fmt(val, scale, decimals=1):
    if val is None:
        return '--'
    return f"{val * scale:.{decimals}f}"


def build_row(name, metric, bench_subdir, key, max_val, scale, models, collected=None):
    cells = []
    for model_dir, _ in models:
        if model_dir is None:
            cells.append('--')
            if collected is not None:
                collected.setdefault(model_dir, []).append(None)
            continue
        path = EVALS_ROOT / model_dir
        if not path.exists():
            cells.append('--')
            if collected is not None:
                collected.setdefault(model_dir, []).append(None)
            continue
        score = get_score(path, bench_subdir, key)
        if score is None:
            cells.append('--')
            if collected is not None:
                collected.setdefault(model_dir, []).append(None)
            continue
        normalized = score / max_val
        cells.append(fmt(normalized, scale, decimals=1))
        if collected is not None:
            collected.setdefault(model_dir, []).append(normalized * scale)
    return f"{name:<16} & {metric:<22} & " + " & ".join(cells) + r" \\"


def build_mean_row(collected, models):
    cells = []
    for model_dir, _ in models:
        vals = [v for v in collected.get(model_dir, []) if v is not None]
        if not vals:
            cells.append('--')
        else:
            cells.append(f"{np.mean(vals):.1f}")
    return f"{'Overall Mean':<16} & {'':<22} & " + " & ".join(cells) + r" \\"


def render_table(models):
    header_models = [pretty for _, pretty in models]
    col_spec = "l l|" + "c" * len(models)
    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append("Task & Metric & " + " & ".join(header_models) + r" \\")
    lines.append(r"\midrule")
    n_cols = 2 + len(models)
    collected = {}
    lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\textbf{Multiple Choice}} \\")
    for row in MULTIPLE_CHOICE_BENCHMARKS:
        lines.append(build_row(*row, models=models, collected=collected))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\textbf{Open-ended}} \\")
    for row in OPEN_ENDED_BENCHMARKS:
        lines.append(build_row(*row, models=models, collected=collected))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\textbf{Compositionality}} \\")
    for row in COMP_BENCHMARKS:
        lines.append(build_row(*row, models=models, collected=collected))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\textbf{Captioning}} \\")
    for row in CAP_BENCHMARKS:
        lines.append(build_row(*row, models=models, collected=collected))
    lines.append(r"\midrule")
    lines.append(build_mean_row(collected, models))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def make_noimage_models():
    out = []
    for model_dir, pretty in MODELS:
        if model_dir is None:
            out.append((None, pretty))
            continue
        candidate = f"{model_dir}_noimage"
        if (EVALS_ROOT / candidate).exists():
            out.append((candidate, pretty))
        else:
            out.append((None, pretty))
    return out


def main():
    print("% === Table: with image ===")
    print(render_table(MODELS))
    print()
    print("% === Table: no image ===")
    print(render_table(make_noimage_models()))


if __name__ == '__main__':
    main()
