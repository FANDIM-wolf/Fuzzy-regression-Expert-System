"""Командный интерфейс: python -m career_es <команда>."""

from __future__ import annotations

import argparse
import json
import sys

from .expert import ExpertSystem, filter_by_preferences, load_candidates
from .model import CareerModel, optimize, train


def _print_metrics(metrics: dict) -> None:
    print(f"{'Целевая переменная':<24}{'R^2':>9}{'MAE':>12}{'MAE (среднее)':>16}")
    for name, m in metrics.items():
        print(
            f"{name:<24}{m['r2']:>9.4f}{m['mae']:>12.3f}{m['baseline_mae']:>16.3f}"
        )
    worse = [n for n, m in metrics.items() if m["r2"] <= 0]
    if worse:
        print()
        print(
            "Внимание: R^2 <= 0 для "
            + ", ".join(worse)
            + " — модель не точнее предсказания среднего значения."
        )


def cmd_train(args) -> int:
    params = optimize(args.data, n_trials=args.trials) if args.optimize else None
    if params:
        print(f"Лучшие гиперпараметры: {params}")
    model, metrics = train(args.data, params=params, test_size=args.test_size)
    model.save(args.model)
    print(f"Модель сохранена: {args.model}\n")
    _print_metrics(metrics)
    return 0


def cmd_evaluate(args) -> int:
    from sklearn.model_selection import train_test_split

    from .data import load_dataset, split_xy
    from .model import evaluate

    model = CareerModel.load(args.model)
    X, y = split_xy(load_dataset(args.data))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=42)
    _print_metrics(evaluate(model, X_te, y_te, X_tr, y_tr))
    return 0


def cmd_predict(args) -> int:
    system = ExpertSystem.from_file(args.model, hybrid_weight=args.hybrid_weight)
    features = json.loads(args.features) if args.features else json.load(sys.stdin)
    result = system.predict_one(**features)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_recommend(args) -> int:
    system = ExpertSystem.from_file(args.model, hybrid_weight=args.hybrid_weight)
    candidates = load_candidates(args.input)
    enriched = system.predict_frame(candidates)

    preferences = json.loads(args.preferences) if args.preferences else {}
    selected = filter_by_preferences(enriched, preferences)

    enriched.to_csv(args.output, index=False)
    print(f"Все прогнозы: {args.output} ({len(enriched)} строк)")
    if args.filtered:
        selected.to_csv(args.filtered, index=False)
        print(f"Отфильтровано: {args.filtered} ({len(selected)} строк)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="career_es")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="обучить модель")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="career_model.joblib")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--optimize", action="store_true")
    p.add_argument("--trials", type=int, default=20)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="оценить сохранённую модель")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="career_model.joblib")
    p.add_argument("--test-size", type=float, default=0.2)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="прогноз для одного студента (JSON)")
    p.add_argument("--model", default="career_model.joblib")
    p.add_argument("--features", help='JSON, например \'{"Age": 22, ...}\'')
    p.add_argument("--hybrid-weight", type=float, default=0.5)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("recommend", help="прогноз и фильтрация для CSV кандидатов")
    p.add_argument("--model", default="career_model.joblib")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="predictions.csv")
    p.add_argument("--filtered")
    p.add_argument("--preferences", help='JSON, например \'{"Career_Satisfaction": ["high"]}\'')
    p.add_argument("--hybrid-weight", type=float, default=0.5)
    p.set_defaults(func=cmd_recommend)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
