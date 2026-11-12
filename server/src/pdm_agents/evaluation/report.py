"""Write an experiment report: metrics.json, CSV + LaTeX tables, PNG plots, resolved config."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pdm_agents.logging_setup import get_logger

log = get_logger(__name__)

KEY_SYSTEMS = ["lgbm_global", "sequence", "multi_agent_no_rag", "full_system", "majority_vote",
               "no_calibration"]


def _clean(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, np.floating):
        return _clean(float(v))
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_clean(x) for x in v]
    return v


def _latex(df: pd.DataFrame, caption: str) -> str:
    def cell(x: Any) -> str:
        if isinstance(x, float):
            return "--" if math.isnan(x) else f"{x:.4g}"
        return str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    cols = [cell(c) for c in df.columns]
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", r"\hline",
             " & ".join(cols) + r" \\", r"\hline"]
    lines += [" & ".join(cell(x) for x in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\hline", r"\end{tabular}", rf"\caption{{{cell(caption)}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _write_table(root: Path, name: str, df: pd.DataFrame, caption: str) -> None:
    (root / "tables").mkdir(parents=True, exist_ok=True)
    with (root / "tables" / f"{name}.csv").open("w", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False)
    (root / "tables" / f"{name}.tex").write_text(_latex(df, caption), encoding="utf-8")


def _save(fig, root: Path, name: str) -> None:
    (root / "plots").mkdir(parents=True, exist_ok=True)
    with (root / "plots" / f"{name}.png").open("wb") as fh:
        fig.savefig(fh, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _ms(df: pd.DataFrame, col: str) -> pd.Series:
    """'mean ± std' strings for display tables."""
    mean, std = df[f"{col}_mean"], df[f"{col}_std"]
    return pd.Series([("--" if pd.isna(m) else f"{m:.4f}" + ("" if pd.isna(s) else f" ± {s:.4f}"))
                      for m, s in zip(mean, std, strict=True)], index=df.index)


def write_report(out: Path, spec, runner, result: dict) -> None:
    cfg = runner.cfg
    h = cfg.max_horizon
    r0 = cfg.decision.default_cost_ratio
    out.mkdir(parents=True, exist_ok=True)
    agg: pd.DataFrame = result["summary"]["aggregate"]
    sig: pd.DataFrame = result["summary"]["significance"]
    results: pd.DataFrame = result["results"]

    # ---------------------------------------------------------------- tables
    for set_name in spec.sets:
        a = agg[agg["set"] == set_name].reset_index(drop=True)
        if a.empty:
            continue
        pred = pd.DataFrame({"system": a["system"]})
        for hh in cfg.horizons:
            pred[f"AUPRC {hh}d"] = _ms(a, f"auprc_{hh}")
        pred[f"ROC-AUC {h}d"] = _ms(a, f"roc_auc_{h}")
        for far in cfg.evaluation.far_levels:
            pred[f"recall@{far:g}FAR {h}d"] = _ms(a, f"recall@far{far:g}_{h}")
        pred[f"ECE {h}d"] = _ms(a, f"ece_{h}")
        pred["detected"] = _ms(a, "lead_detected_share")
        pred["lead days"] = _ms(a, "lead_mean_lead_days")
        _write_table(out, f"prediction_{set_name}", pred,
                     f"Prediction metrics on the {set_name} set (mean ± std over seeds "
                     f"{cfg.seeds}); recall at fixed false-alarm rate; lead time at "
                     f"{cfg.evaluation.far_levels[0]:g} FAR.")
        raw_cols = ["system"] + [c for c in a.columns if c.endswith("_mean") or
                                 c.endswith("_std")]
        _write_table(out, f"raw_{set_name}", a[raw_cols], f"All metrics ({set_name})")
        cost_rows = []
        for ratio in cfg.decision.cost_ratios:
            for _, row in a.iterrows():
                cost_rows.append({
                    "system": row["system"], "cost_ratio": ratio,
                    "cost (escalation: model)": row[f"cost_model_r{ratio:g}_mean"],
                    "cost (escalation: oracle)": row[f"cost_oracle_r{ratio:g}_mean"],
                    "cost (no escalation)": row[f"cost_noescalation_r{ratio:g}_mean"],
                    "escalation rate": row[f"escalation_rate_r{ratio:g}_mean"],
                    "always monitor": row[f"cost_always_monitor_r{ratio:g}_mean"],
                    "hindsight optimum": row[f"cost_hindsight_r{ratio:g}_mean"]})
        _write_table(out, f"cost_{set_name}", pd.DataFrame(cost_rows),
                     f"Mean realized cost per drive-window ({set_name}); 1 = one planned "
                     "replacement. Escalated cases add the review cost to either the model's "
                     "suggested action (model) or the hindsight-best action (oracle).")
    if not sig.empty:
        _write_table(out, "significance", sig,
                     f"AUPRC ({h}d) and cost (ratio {r0:g}) with 95% cluster-bootstrap CIs "
                     f"(by drive, {cfg.evaluation.bootstrap_samples} resamples, seed "
                     f"{cfg.seeds[0]}); paired bootstrap p-values vs the reference (best "
                     "baseline, or full_system in ablation-only runs).")
    sel = pd.concat([df.assign(key=k) for k, df in result["selective"].items()]) \
        if result["selective"] else pd.DataFrame()
    if not sel.empty:
        sel[["set", "system"]] = sel["key"].str.split(":", expand=True)
        _write_table(out, "selective", sel.drop(columns="key"),
                     "Selective accuracy (decision equals the hindsight-optimal action) vs "
                     "coverage, ranking cases by 1 - uncertainty.")
    for set_name, dtab in result["drift"].items():
        _write_table(out, f"drift_{set_name}", dtab,
                     f"Monthly drift (PSI vs validation period) and recalibration ({set_name}).")
    for set_name, cols in (result.get("predictions") or {}).items():
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        with (out / "predictions" / f"{set_name}.parquet").open("wb") as fh:
            pd.DataFrame(cols).to_parquet(fh, index=False, compression="zstd")
    llm = result["llm"]
    if llm is not None:
        _write_table(out, "llm_sample", llm["table"],
                     f"Graph runs on a stratified sample of {llm['n']} test windows "
                     f"({llm['positives']} positive at {h}d); LLM mode: {llm['mode']}.")

    # ----------------------------------------------------------------- plots
    for set_name in spec.sets:
        curves = {k.split(":")[1]: v for k, v in result["pr"].items()
                  if k.startswith(f"{set_name}:") and k.split(":")[1] in KEY_SYSTEMS}
        if curves:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for name, (rec, prec) in curves.items():
                ax.plot(rec, prec, label=name, lw=1.4)
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.set_title(f"Precision-recall, {h}-day horizon ({set_name}, seed {cfg.seeds[0]})")
            ax.legend(fontsize=8)
            _save(fig, out, f"pr_{set_name}")
        rel = {k.split(":")[1]: v for k, v in result["reliability"].items()
               if k.startswith(f"{set_name}:") and k.split(":")[1] in KEY_SYSTEMS}
        if rel:
            fig, ax = plt.subplots(figsize=(5, 5))
            lim = max(max(d["mean_predicted"].max(), d["observed_rate"].max())
                      for d in rel.values())
            ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="perfect")
            for name, d in rel.items():
                ax.plot(d["mean_predicted"], d["observed_rate"], "o-", ms=3, label=name)
            ax.set_xscale("symlog", linthresh=1e-3)
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_xlabel("mean predicted probability")
            ax.set_ylabel("observed failure rate")
            ax.set_title(f"Reliability, {h}-day ({set_name})")
            ax.legend(fontsize=7)
            _save(fig, out, f"reliability_{set_name}")
        a = agg[agg["set"] == set_name]
        if not a.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for _, row in a[a["system"].isin(KEY_SYSTEMS)].iterrows():
                ys = [row[f"cost_model_r{r:g}_mean"] for r in cfg.decision.cost_ratios]
                ax.plot(cfg.decision.cost_ratios, ys, "o-", label=row["system"])
            first = a.iloc[0]
            ax.plot(cfg.decision.cost_ratios,
                    [first[f"cost_always_monitor_r{r:g}_mean"] for r in cfg.decision.cost_ratios],
                    "k:", label="always monitor")
            ax.plot(cfg.decision.cost_ratios,
                    [first[f"cost_hindsight_r{r:g}_mean"] for r in cfg.decision.cost_ratios],
                    "k--", label="hindsight optimum")
            ax.set_xscale("log")
            ax.set_xlabel("cost ratio (missed failure / planned replacement)")
            ax.set_ylabel("mean realized cost per window")
            ax.set_title(f"Cost sweep ({set_name})")
            ax.legend(fontsize=7)
            _save(fig, out, f"cost_{set_name}")
            fig, ax = plt.subplots(figsize=(7, 0.35 * len(a) + 1.2))
            order = a.sort_values(f"auprc_{h}_mean")
            ax.barh(order["system"], order[f"auprc_{h}_mean"],
                    xerr=order[f"auprc_{h}_std"].fillna(0), color="#475569")
            ax.set_xlabel(f"AUPRC ({h}-day), mean ± std over seeds")
            ax.set_title(f"Systems and ablations ({set_name})")
            _save(fig, out, f"auprc_{set_name}")
    if not sel.empty:
        for set_name in spec.sets:
            s = sel[(sel["set"] == set_name) & sel["system"].isin(KEY_SYSTEMS)]
            if s.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for name, d in s.groupby("system"):
                ax.plot(d["coverage"], d["selective_accuracy"], "o-", label=name)
            ax.set_xlabel("coverage")
            ax.set_ylabel("selective accuracy")
            ax.set_title(f"Selective accuracy vs coverage ({set_name})")
            ax.legend(fontsize=7)
            _save(fig, out, f"selective_{set_name}")

    # ------------------------------------------------------- metrics.json etc.
    headline: dict[str, Any] = {}
    test_agg = agg[agg["set"] == spec.sets[0]].set_index("system")
    for name in test_agg.index:
        headline[name] = {
            f"auprc_{h}": test_agg.loc[name, f"auprc_{h}_mean"],
            f"auprc_{h}_std": test_agg.loc[name, f"auprc_{h}_std"],
            f"recall@{cfg.evaluation.far_levels[0]:g}far_{h}":
                test_agg.loc[name, f"recall@far{cfg.evaluation.far_levels[0]:g}_{h}_mean"],
            f"cost_r{r0:g}": test_agg.loc[name, f"cost_model_r{r0:g}_mean"],
        }
    notes = [
        f"Sets: {', '.join(spec.sets)}; seeds {cfg.seeds}; evaluation on the natural class "
        "distribution (uniform sample capped at "
        f"{cfg.data.eval_max_windows} windows per set).",
        "Costs: 1 unit = one planned replacement; a missed failure costs the cost ratio. "
        "'model' escalation outcome = the reviewer applies the policy's suggested action "
        "(conservative); 'oracle' = the reviewer picks the hindsight-best action (optimistic).",
        "single_llm_agent uses the all-features model for its risk, so its numeric metrics "
        "equal lgbm_global; no_knowledge_retrieval equals full_system numerically.",
    ]
    if llm is not None:
        notes.append(f"Graph runs: stratified sample of {llm['n']} test windows "
                     f"({llm['positives']} positives), LLM mode '{llm['mode']}'. {llm['note']}")
    metrics = {
        "id": out.name, "name": spec.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "systems": list(runner.systems), "sets": spec.sets, "seeds": cfg.seeds,
        "horizons": cfg.horizons, "cost_ratios": cfg.decision.cost_ratios,
        "headline": headline,
        "descriptions": {n: d.description + (f" Note: {d.note}" if d.note else "")
                         for n, d in runner.systems.items()},
        "notes": "\n".join(notes),
        "elapsed_s": round(result["elapsed_s"], 1),
        "per_seed": results.to_dict(orient="records"),
        "aggregate": agg.to_dict(orient="records"),
        "significance": sig.to_dict(orient="records"),
        "llm_sample": None if llm is None else {
            "mode": llm["mode"], "n": llm["n"], "positives": llm["positives"],
            "note": llm["note"], "table": llm["table"].to_dict(orient="records")},
    }
    (out / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2), encoding="utf-8")
    resolved = json.loads(cfg.model_dump_json(by_alias=True))
    if resolved.get("llm", {}).get("api_key"):
        resolved["llm"]["api_key"] = "***"  # never write secrets into reports
    (out / "config.yaml").write_text(yaml.safe_dump(
        {"experiment": spec.__dict__, "overrides": runner.overrides, "resolved": resolved},
        sort_keys=False), encoding="utf-8")
    log.info("report written to %s", out)
