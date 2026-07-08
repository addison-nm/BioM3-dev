#!/usr/bin/env python3
"""Reference-free head-to-head: does each model rank FUNCTIONAL SH3 designs above
NONFUNCTIONAL ones in absolute likelihood?

For each model we score every design's length-normalized log P(seq | z_c) with the
(fixed) ProteoScribeLikelihoodEstimator (mode b: <START>/PAD context, residues+END
query), per prompt, then report the functional-vs-nonfunctional AUC = P(score_func
> score_nonfunc) = the reference-free 'abs_acc'. z_c comes from the same frozen
run1_base PenCL+Facilitator all runs used.

Models compared:
  BASE     run1 base ProteoScribe (generic)
  FT       SH3 full-finetune (un-aligned)
  DPO_run1 run1-base + DPO (align-only)
  DPO_ft   SH3-finetune + DPO (finetune-then-align)
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.Stage3.io import build_model_ProteoScribe
from biom3.Stage3.tools import ProteoScribeLikelihoodEstimator, LikelihoodConfig
from biom3.rl.io import load_pencl_frozen, load_facilitator_frozen
from biom3.rl.grpo import _PromptEncoder
from biom3.backend.device import get_device, setup_logger

logger = setup_logger("eval_ranking")

MODELS = {
    "BASE":     "weights/ProteoScribe/run1_base_proteoscribe.ckpt",
    "FT":       "weights/ProteoScribe/sh3_ft_prod.pth",
    "DPO_run1": "outputs/dpo/dpo_sh3_paired_v2/final.pt",
    "DPO_ft":   "outputs/dpo/dpo_sh3_ftbase_d70/final.pt",
}
S3_CONFIG = "configs/inference/stage3_ProteoScribe_sample.json"
LCFG = LikelihoodConfig(n_quadrature=6, n_repeats=2, seed=0)


def load_estimator(path, cfg, device):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:      # DPO final.pt
        model = build_model_ProteoScribe(cfg)
        model.load_state_dict(obj["model_state"], strict=True)
        return ProteoScribeLikelihoodEstimator(model, cfg, device=device)
    return ProteoScribeLikelihoodEstimator.from_weights(cfg, path, device=device)


def main():
    device = get_device()
    cfg = convert_to_namespace(load_json_config(S3_CONFIG)); cfg.device = str(device)

    # Frozen z_c front-end (same run1_base encoders every run used).
    cfg1 = convert_to_namespace(load_json_config("configs/inference/stage1_PenCL.json"))
    cfg2 = convert_to_namespace(load_json_config("configs/inference/stage2_Facilitator.json"))
    s1 = load_pencl_frozen(cfg1, "weights/PenCL/run1_base_pencl.ckpt", device=str(device))
    s2 = load_facilitator_frozen(cfg2, "weights/Facilitator/run1_base_facilitator.ckpt", device=str(device))
    encode = _PromptEncoder(s1, s2, cfg1, device)

    df = pd.read_csv("data/rl/processed/biom3_designs.csv")
    prompts = sorted(p for p in df["prompt"].dropna().unique()
                     if ((df.prompt == p) & (df.functional == 1)).any()
                     and ((df.prompt == p) & (df.functional == 0)).any())
    z_by_prompt = {}
    for p in prompts:
        cap = df[df.prompt == p]["prompt_text"].iloc[0]
        z_by_prompt[p] = encode(cap).detach()
    logger.info("Usable prompts (both classes): %s", prompts)

    results = {}
    for name, path in MODELS.items():
        logger.info("Scoring model %s (%s)...", name, path)
        est = load_estimator(path, cfg, device)
        per_prompt_auc, all_func, all_nonf = {}, [], []
        for p in prompts:
            z = z_by_prompt[p]
            sub = df[df.prompt == p]
            func, nonf = [], []
            for r in sub.itertuples():
                try:
                    res = est.estimate(r.sequence, z, config=LCFG)
                    score = res.log_likelihood / max(res.n_query, 1)   # per-residue logP
                except Exception:
                    continue
                (func if r.functional == 1 else nonf).append(score)
            func, nonf = np.array(func), np.array(nonf)
            if len(func) and len(nonf):
                U = mannwhitneyu(func, nonf, alternative="two-sided").statistic
                per_prompt_auc[int(p)] = U / (len(func) * len(nonf))
                all_func.append(func); all_nonf.append(nonf)
        F, N = np.concatenate(all_func), np.concatenate(all_nonf)
        overall = mannwhitneyu(F, N, alternative="two-sided").statistic / (len(F) * len(N))
        results[name] = dict(overall=overall, per_prompt=per_prompt_auc,
                             func_mean=F.mean(), nonf_mean=N.mean(), margin=F.mean() - N.mean())
        del est
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n==== Functional-vs-nonfunctional AUC (abs_acc): P(logP_func > logP_nonfunc) ====")
    hdr = "model      | overall | " + " | ".join(f"p{p}" for p in prompts) + " | func_lp  nonf_lp  margin"
    print(hdr)
    for name, r in results.items():
        pp = " | ".join(f"{r['per_prompt'].get(p, float('nan')):.3f}" for p in prompts)
        print("%-10s | %.3f   | %s | %+.3f  %+.3f  %+.3f"
              % (name, r["overall"], pp, r["func_mean"], r["nonf_mean"], r["margin"]))
    print("\n(0.5 = no discrimination; 1.0 = always ranks functional above nonfunctional)")


if __name__ == "__main__":
    main()
