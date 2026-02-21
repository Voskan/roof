import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Mine hard examples from per-sample metrics CSV')
    parser.add_argument('--metrics-csv', required=True, help='CSV path from tools/error_analysis.py')
    parser.add_argument('--out-file', required=True, help='Output text file with hard sample ids')
    parser.add_argument('--top-k', type=int, default=200, help='Number of hardest samples to keep')
    parser.add_argument('--min-hard-score', type=float, default=0.0, help='Optional minimum hard score')
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.metrics_csv)
    out_path = Path(args.out_file)
    if not csv_path.exists():
        raise FileNotFoundError(f'Metrics CSV not found: {csv_path}')

    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get('sample_id', '').strip()
            if not sid:
                continue
            try:
                hard = float(row.get('hard_score', 0.0))
            except Exception:
                continue
            if hard < float(args.min_hard_score):
                continue
            rows.append((sid, hard))

    rows.sort(key=lambda x: x[1], reverse=True)
    if args.top_k > 0:
        rows = rows[: int(args.top_k)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join([r[0] for r in rows]) + ('\n' if rows else ''), encoding='utf-8')
    print(f'Wrote {len(rows)} hard example ids to {out_path}')


if __name__ == '__main__':
    main()
