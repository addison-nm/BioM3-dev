#!/usr/bin/env python3
"""Train a surrogate regressor for the GRPO surrogate-in-the-loop reward.

Reads a (sequence, scalar) TSV — typically the output of
``make_grpo_synthetic_eval.py``, or in real use a wet-lab assay file —
featurizes the sequences (one-hot or ESM-2 mean-pool), fits a sklearn
regressor (Ridge by default; MLP optional), reports R²/MAE/RMSE on
held-out test rows, and saves the fitted estimator next to a small JSON
sidecar describing the featurizer config so the eval / GRPO path can
reconstruct the pipeline.

Outputs
-------
``--output_path``               (default ``outputs/grpo/surrogate.joblib``)
``<output_path>.config.json``   (featurizer + model config sidecar)

Reload at GRPO time
-------------------
    import joblib, json
    from biom3.rl.featurizers import build_featurizer
    from biom3.rl.rewards import SurrogateReward

    predictor = joblib.load("outputs/grpo/surrogate.joblib")
    cfg = json.load(open("outputs/grpo/surrogate.joblib.config.json"))
    featurizer = build_featurizer(cfg["featurizer"], **cfg["featurizer_kwargs"])
    reward_fn = SurrogateReward(predictor=predictor, featurizer=featurizer)

Example
-------
    python scripts/train_grpo_surrogate.py \\
        --train_tsv data/datasets/SH3/synthetic_train.tsv \\
        --test_tsv  data/datasets/SH3/synthetic_test.tsv \\
        --featurizer onehot --model ridge \\
        --output_path outputs/grpo/surrogate_aa.joblib
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from biom3.rl.featurizers import build_featurizer


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train_tsv", required=True)
    p.add_argument("--test_tsv", required=True)
    p.add_argument("--sequence_column", default="protein_sequence")
    p.add_argument("--value_column", default="functional_score")
    p.add_argument("--featurizer", choices=["onehot", "esm2"], default="onehot")
    p.add_argument("--model", choices=["ridge", "mlp"], default="ridge")
    p.add_argument("--output_path", default="outputs/grpo/surrogate.joblib")

    # Featurizer kwargs
    p.add_argument("--max_length", type=int, default=256,
                   help="Pad/truncate target length for both featurizers.")
    p.add_argument("--esm_model_path",
                   default="./weights/LLMs/esm2_t33_650M_UR50D.pt",
                   help="ESM-2 weights path (only used when --featurizer esm2).")
    p.add_argument("--esm_layer", type=int, default=33)
    p.add_argument("--esm_batch_size", type=int, default=16)
    p.add_argument("--device", default=None,
                   help="Device for ESM-2 featurizer (cuda|xpu|cpu). Default: auto.")

    # Model kwargs
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Ridge regularization (only used when --model ridge).")
    p.add_argument("--mlp_hidden", default="256,128",
                   help="Comma-separated MLP hidden sizes (only used when --model mlp).")
    p.add_argument("--mlp_max_iter", type=int, default=200)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _build_featurizer(args):
    if args.featurizer == "onehot":
        return (
            build_featurizer("onehot", max_length=args.max_length),
            {"max_length": args.max_length},
        )
    # esm2
    return (
        build_featurizer(
            "esm2",
            model_path=args.esm_model_path,
            layer=args.esm_layer,
            device=args.device,
            max_length=args.max_length,
            batch_size=args.esm_batch_size,
        ),
        {
            "model_path": args.esm_model_path,
            "layer": args.esm_layer,
            "device": args.device,
            "max_length": args.max_length,
            "batch_size": args.esm_batch_size,
        },
    )


def _build_model(args):
    if args.model == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=args.alpha, random_state=args.seed)
    # mlp
    from sklearn.neural_network import MLPRegressor
    hidden = tuple(int(x) for x in args.mlp_hidden.split(","))
    return MLPRegressor(
        hidden_layer_sizes=hidden,
        max_iter=args.mlp_max_iter,
        random_state=args.seed,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)

    df_train = pd.read_csv(args.train_tsv, sep="\t")
    df_test = pd.read_csv(args.test_tsv, sep="\t")
    for df, label in ((df_train, "train"), (df_test, "test")):
        for col in (args.sequence_column, args.value_column):
            if col not in df.columns:
                raise KeyError(f"{label} TSV missing column {col!r}; got {list(df.columns)}")

    print(
        f"[info] train={len(df_train)} test={len(df_test)} "
        f"value_column={args.value_column}",
        flush=True,
    )

    featurizer, featurizer_kwargs = _build_featurizer(args)
    t0 = time.time()
    X_train = featurizer(df_train[args.sequence_column].tolist())
    X_test = featurizer(df_test[args.sequence_column].tolist())
    y_train = df_train[args.value_column].to_numpy(dtype=np.float64)
    y_test = df_test[args.value_column].to_numpy(dtype=np.float64)
    print(
        f"[info] Featurized in {time.time() - t0:.1f}s | "
        f"X_train={X_train.shape} X_test={X_test.shape}",
        flush=True,
    )

    model = _build_model(args)
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"[info] Fit {args.model} in {time.time() - t0:.1f}s", flush=True)

    train_metrics = _metrics(y_train, model.predict(X_train))
    test_metrics = _metrics(y_test, model.predict(X_test))
    print(
        f"[info] train R²={train_metrics['r2']:.4f} "
        f"MAE={train_metrics['mae']:.3f} RMSE={train_metrics['rmse']:.3f}",
        flush=True,
    )
    print(
        f"[info] test  R²={test_metrics['r2']:.4f} "
        f"MAE={test_metrics['mae']:.3f} RMSE={test_metrics['rmse']:.3f}",
        flush=True,
    )

    import joblib
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    joblib.dump(model, args.output_path)
    print(f"[info] Saved estimator -> {args.output_path}", flush=True)

    sidecar = {
        "featurizer": args.featurizer,
        "featurizer_kwargs": featurizer_kwargs,
        "model": args.model,
        "sequence_column": args.sequence_column,
        "value_column": args.value_column,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "seed": args.seed,
    }
    sidecar_path = args.output_path + ".config.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[info] Sidecar -> {sidecar_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
