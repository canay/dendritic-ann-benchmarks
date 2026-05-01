from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-root', type=str, default='./runs')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.runs_root)
    rows = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        summary = dataset_dir / 'summary_by_model.csv'
        if not summary.exists():
            continue
        with summary.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['dataset'] = dataset_dir.name
                rows.append(row)
    if not rows:
        print('No summary files found.')
        return
    grouped = {}
    for row in rows:
        grouped.setdefault(row['model_name'], []).append(row)

    out = []
    for model_name, items in grouped.items():
        out.append({
            'model_name': model_name,
            'datasets_covered': len(items),
            'mean_best_test_acc': mean(float(x['best_test_acc_mean']) for x in items),
            'mean_accuracy_efficiency': mean(float(x['accuracy_efficiency_mean']) for x in items),
            'mean_loss_efficiency': mean(float(x['loss_efficiency_mean']) for x in items),
        })
    out = sorted(out, key=lambda x: (-x['mean_best_test_acc'], -x['mean_accuracy_efficiency']))
    out_path = root / 'aggregate_across_datasets.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
