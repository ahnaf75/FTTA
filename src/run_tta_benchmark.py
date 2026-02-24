import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

from ftta.engine import run_model_train


DEFAULT_DATASETS = ["anes", "assistments", "heloc", "diabetes_readmission"]
DEFAULT_MODELS = ["mlp", "tabtransformer", "ft_transformer"]
DEFAULT_METHODS = ["ftta", "tent"]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--tent_lr", type=float, default=1e-3)
    parser.add_argument("--tent_steps", type=int, default=1)
    parser.add_argument("--tent_eps", type=float, default=1e-8)
    parser.add_argument("--report_dir", default="reports/tta_benchmark")
    return parser.parse_args()


def _aggregate(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = {}
    for row in rows:
        key = (row["dataset"], row["model"], row["method"])
        grouped.setdefault(key, []).append(row)

    output = []
    for (dataset, model, method), group_rows in grouped.items():
        ood = [float(r["ood_test"]) for r in group_rows if "ood_test" in r]
        unadapt = [float(r["unadapt_ood_test"]) for r in group_rows if "unadapt_ood_test" in r]
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "num_runs": len(group_rows),
                "ood_test_mean": mean(ood) if ood else None,
                "ood_test_std": pstdev(ood) if len(ood) > 1 else 0.0 if ood else None,
                "unadapt_ood_test_mean": mean(unadapt) if unadapt else None,
                "unadapt_ood_test_std": pstdev(unadapt) if len(unadapt) > 1 else 0.0 if unadapt else None,
            }
        )
    return output


def _write_csv(path: Path, rows: List[Dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        values = [str(row.get(k, "")) for k in keys]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_method_delta(aggregate_rows: List[Dict]) -> List[Dict]:
    keyed: Dict[tuple, Dict[str, Dict]] = {}
    for row in aggregate_rows:
        key = (row["dataset"], row["model"])
        keyed.setdefault(key, {})
        keyed[key][row["method"]] = row

    deltas: List[Dict] = []
    for (dataset, model), methods in keyed.items():
        if "ftta" not in methods or "tent" not in methods:
            continue
        ftta = methods["ftta"]
        tent = methods["tent"]
        ftta_mean = ftta.get("ood_test_mean")
        tent_mean = tent.get("ood_test_mean")
        if ftta_mean is None or tent_mean is None:
            continue
        deltas.append(
            {
                "dataset": dataset,
                "model": model,
                "tent_minus_ftta_ood_test_mean": float(tent_mean) - float(ftta_mean),
                "ftta_unadapt_ood_test_mean": ftta.get("unadapt_ood_test_mean"),
                "tent_unadapt_ood_test_mean": tent.get("unadapt_ood_test_mean"),
            }
        )
    return deltas


def main():
    args = _parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict] = []
    failed_rows: List[Dict] = []
    started_at = time.time()

    for dataset in args.datasets:
        for model in args.models:
            for method in args.methods:
                print(f"[RUN] dataset={dataset}, model={model}, method={method}")
                t0 = time.time()
                try:
                    result = run_model_train(
                        experiment=dataset,
                        cache_dir=args.cache_dir,
                        model=model,
                        debug=False,
                        tta_method=method,
                        num_seeds=args.num_seeds,
                        seed_start=args.seed_start,
                        tent_lr=args.tent_lr,
                        tent_steps=args.tent_steps,
                        tent_eps=args.tent_eps,
                    )
                except Exception as exc:
                    failed_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "method": method,
                            "error": str(exc),
                        }
                    )
                    print(f"[FAIL] dataset={dataset}, model={model}, method={method}, error={exc}")
                    continue
                elapsed = time.time() - t0
                for metric in result.get("seed_metrics", []):
                    row = {
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "runtime_sec": elapsed,
                    }
                    row.update(metric)
                    per_seed_rows.append(row)

    aggregate_rows = _aggregate(per_seed_rows)
    delta_rows = _build_method_delta(aggregate_rows)

    per_seed_json = report_dir / "tta_per_seed.json"
    aggregate_json = report_dir / "tta_aggregate.json"
    per_seed_csv = report_dir / "tta_per_seed.csv"
    aggregate_csv = report_dir / "tta_aggregate.csv"
    delta_json = report_dir / "tta_method_delta.json"
    delta_csv = report_dir / "tta_method_delta.csv"
    failures_json = report_dir / "tta_failures.json"
    failures_csv = report_dir / "tta_failures.csv"

    per_seed_json.write_text(json.dumps(per_seed_rows, indent=2), encoding="utf-8")
    aggregate_json.write_text(json.dumps(aggregate_rows, indent=2), encoding="utf-8")
    delta_json.write_text(json.dumps(delta_rows, indent=2), encoding="utf-8")
    failures_json.write_text(json.dumps(failed_rows, indent=2), encoding="utf-8")
    _write_csv(per_seed_csv, per_seed_rows)
    _write_csv(aggregate_csv, aggregate_rows)
    _write_csv(delta_csv, delta_rows)
    _write_csv(failures_csv, failed_rows)

    print(f"Benchmark complete in {time.time() - started_at:.2f}s")
    print(f"Saved: {per_seed_json}")
    print(f"Saved: {aggregate_json}")
    print(f"Saved: {per_seed_csv}")
    print(f"Saved: {aggregate_csv}")
    print(f"Saved: {delta_json}")
    print(f"Saved: {delta_csv}")
    print(f"Saved: {failures_json}")
    print(f"Saved: {failures_csv}")


if __name__ == "__main__":
    main()
