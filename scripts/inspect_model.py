"""Inspect a saved model's configuration and details.
Usage:
  python3 scripts/inspect_model.py                  # list all models
  python3 scripts/inspect_model.py optA_xgboost     # detail one model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.models import load_model  # noqa: E402

MODELS_DIR = REPO_ROOT / "outputs" / "models"


def _file_sizes(d: Path) -> str:
    parts = []
    for f in sorted(d.iterdir()):
        if f.is_file():
            kb = f.stat().st_size / 1024
            parts.append(f"{f.name} ({kb:.1f} KB)")
    return ", ".join(parts)


def _inspect_xgb_single(model, cfg: dict) -> None:
    print(f"  Algorithm:       XGBoost (single output)")
    print(f"  Trees:           {cfg['n_estimators']}, max depth {cfg['max_depth']}, lr {cfg['learning_rate']}")
    booster = model.model.get_booster()
    fmap = booster.get_score(importance_type="gain")
    if fmap and cfg.get("feature_names"):
        names = cfg["feature_names"]
        normalized = {names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(names) else k: v for k, v in fmap.items()}
        ordered = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)[:10]
        total = sum(v for _, v in ordered) or 1.0
        print(f"  Top 10 features by gain:")
        for k, v in ordered:
            print(f"    {k:<20} {v / total * 100:>5.1f}%")


def _inspect_xgb_multi(model, cfg: dict) -> None:
    print(f"  Algorithm:       XGBoost (multi output, one estimator per stat)")
    print(f"  # estimators:    {cfg['n_outputs']}")
    print(f"  Trees per est:   {cfg['n_estimators']}, max depth {cfg['max_depth']}, lr {cfg['learning_rate']}")


def _inspect_mlp(model, cfg: dict) -> None:
    layers = [cfg["input_dim"], *cfg["hidden_layer_sizes"], cfg["output_dim"]]
    print(f"  Algorithm:       MLP (multi-layer perceptron)")
    print(f"  Architecture:    {' -> '.join(str(x) for x in layers)}")
    print(f"  Activation:      {cfg['activation']} (hidden), identity (output)")
    print(f"  Total params:    {cfg['total_parameters']:,}")
    if cfg.get("n_iter") is not None:
        print(f"  Trained iters:   {cfg['n_iter']} (early stopping)")
    w = json.loads((MODELS_DIR / cfg.get("_dirname", "") / "weights.json").read_text())
    first_w = np.asarray(w["coefs"][0])
    print(f"  First layer weights: shape {first_w.shape}, mean {first_w.mean():+.4f}, std {first_w.std():.4f}")


def _inspect_autoencoder(model, cfg: dict) -> None:
    print(f"  Algorithm:       Autoencoder + KNN (player-similarity prediction)")
    print(f"  Latent dim:      {cfg['latent_dim']}")
    print(f"  Input dim:       {cfg['input_dim']}")
    print(f"  Encoder layers:  {cfg['input_dim']} -> 32 -> 16 -> {cfg['latent_dim']}")
    print(f"  Decoder layers:  {cfg['latent_dim']} -> 16 -> 32 -> {cfg['input_dim']}")
    print(f"  KNN k:           {cfg['k']}")
    print(f"  Train embeddings: {cfg['n_train']:,} player-seasons in latent space")
    print(f"  Target shape:    {cfg['target_shape']}")


def _inspect_ensemble(model, cfg: dict) -> None:
    print(f"  Algorithm:       Equal-weight average ensemble")
    print(f"  Components:      {cfg['components']}")


_INSPECTORS = {
    "xgboost_single":   _inspect_xgb_single,
    "xgboost_multi":    _inspect_xgb_multi,
    "mlp":              _inspect_mlp,
    "autoencoder_knn":  _inspect_autoencoder,
    "ensemble_avg":     _inspect_ensemble,
}


def _list_all() -> None:
    print("Available models:")
    print(f"  {'name':<22} {'type':<22} files")
    print(f"  {'-'*22} {'-'*22} {'-'*40}")
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir() and (d / "config.json").exists():
            cfg = json.loads((d / "config.json").read_text())
            files = ", ".join(sorted(f.name for f in d.iterdir() if f.is_file()))
            print(f"  {d.name:<22} {cfg.get('type', '?'):<22} {files}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a saved model.")
    ap.add_argument("name", nargs="?", help="Model name (e.g. optA_xgboost). Omit to list all.")
    args = ap.parse_args()

    if not args.name:
        _list_all()
        return

    d = MODELS_DIR / args.name
    if not d.is_dir():
        print(f"No model named '{args.name}' in {MODELS_DIR}", file=sys.stderr)
        sys.exit(1)

    cfg = json.loads((d / "config.json").read_text())
    cfg["_dirname"] = args.name
    kind = cfg["type"]

    print(f"\nModel: {args.name}")
    print(f"  Type:            {kind}")
    print(f"  Saved at:        outputs/models/{args.name}/")
    print(f"  Files:           {_file_sizes(d)}")

    model = load_model(args.name, MODELS_DIR)
    inspector = _INSPECTORS.get(kind)
    if inspector:
        inspector(model, cfg)
    print()


if __name__ == "__main__":
    main()
