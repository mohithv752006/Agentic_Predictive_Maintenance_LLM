"""Run an experiment YAML end to end over several seeds and write a report.

Numeric components are scored on the full (natural-distribution) test sets.
Systems that use the LLM are additionally run through the real LangGraph pipeline
on a stratified sample (all positives up to a cap + random negatives) because
running every test window through the LLM is too slow; the report states the
sample size, its composition and the LLM mode actually used.
"""

from __future__ import annotations

import dataclasses
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pdm_agents.config import AppConfig, deep_merge, load_config
from pdm_agents.data.features import load_split
from pdm_agents.data.labels import label_column
from pdm_agents.data.store import Store
from pdm_agents.decision.cost import realized_costs
from pdm_agents.decision.drift import relevant_level
from pdm_agents.decision.policy import decide_batch
from pdm_agents.evaluation import metrics as M
from pdm_agents.evaluation.ablations import ablations
from pdm_agents.evaluation.baselines import BASELINES, EvalContext, SystemDef, SystemOutput
from pdm_agents.logging_setup import get_logger
from pdm_agents.models.base import logit, sigmoid
from pdm_agents.models.bundle import bundle_dir, load_bundle, monotone_in_horizon
from pdm_agents.rag.case_memory import CaseMemory
from pdm_agents.rag.knowledge_index import index_dir
from pdm_agents.schemas import PredictRequest

log = get_logger(__name__)

EVAL_SETS = {"test": "leadtime_test", "unseen": "leadtime_unseen"}
BASELINE_ORDER = ["lgbm_global", "sequence", "single_llm_agent", "multi_agent_no_rag"]
# systems whose exported predictions keep every horizon (the rest keep the max horizon only)
EXPORT_ALL_HORIZONS = ["full_system", "lgbm_global", "sequence", "multi_agent_no_rag"]
MIN_RECAL_POSITIVES = 5


@dataclass
class ExperimentSpec:
    name: str
    systems: list[str]
    sets: list[str] = field(default_factory=lambda: ["test", "unseen"])
    llm_mode: str = "live"  # live | offline | fake
    run_llm_sample: bool = True
    llm_systems: list[str] | None = None  # default: every listed system with graph overrides
    bootstrap: bool = True
    export_predictions: bool = False  # per-window predictions (seed 0) for notebooks/plots
    require_live: bool = False  # fail instead of falling back to offline graph runs

    @classmethod
    def from_yaml(cls, path: Path) -> tuple[ExperimentSpec, dict]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = cls(**raw.get("experiment", {}))
        return spec, raw.get("overrides", {}) or {}


def system_registry(cfg: AppConfig) -> dict[str, SystemDef]:
    reg = {name: make(cfg) for name, make in BASELINES.items()}
    reg.update({name: make(cfg) for name, make in ablations(cfg).items()})
    return reg


# ------------------------------------------------------------------ helpers


def _labels(df: pd.DataFrame, horizons: list[int]) -> dict[int, np.ndarray]:
    return {h: df[label_column(h)].to_numpy("float64") for h in horizons}


def _drift_recalibrate(ctx: EvalContext, sysdef: SystemDef, table: str,
                       out: SystemOutput) -> tuple[SystemOutput, pd.DataFrame, dict]:
    """Monthly, per drive model: if drift for that month is >= warn, refit a Platt map on
    the most recent windows whose labels had matured before the month began."""
    cfg = ctx.cfg
    df = ctx.df(table)
    val = ctx.df("val")
    val_out = sysdef.score(ctx, "val")
    as_of = pd.to_datetime(df["as_of"])
    month = as_of.dt.to_period("M")
    pool = pd.concat([
        pd.DataFrame({"as_of": pd.to_datetime(val["as_of"]), "model": val["model"].astype(str),
                      **{f"p_{h}": val_out.p[h] for h in ctx.bundle.horizons},
                      **{f"y_{h}": val[label_column(h)].to_numpy() for h in ctx.bundle.horizons}}),
        pd.DataFrame({"as_of": as_of, "model": df["model"].astype(str),
                      **{f"p_{h}": out.p[h] for h in ctx.bundle.horizons},
                      **{f"y_{h}": df[label_column(h)].to_numpy() for h in ctx.bundle.horizons}}),
    ], ignore_index=True)
    p_new = {h: out.p[h].copy() for h in ctx.bundle.horizons}
    rows = []
    maps: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
    for (model, m), idx in df.groupby([df["model"].astype(str), month]).indices.items():
        start = m.start_time
        rep = ctx.bundle.drift.report(df.iloc[idx], model, cfg)
        lo = start - pd.Timedelta(days=cfg.drift.recalibration_window_days + cfg.max_horizon)
        hist = pool[(pool["model"] == model) & (pool["as_of"] >= lo)
                    & (pool["as_of"] + pd.Timedelta(days=cfg.max_horizon) < start)]
        applied = False
        level = relevant_level(rep, ctx.bundle.agent_weights(cfg.max_horizon), cfg)
        if level in ("warn", "alert"):
            for h in ctx.bundle.horizons:
                y = hist[f"y_{h}"].to_numpy("float64")
                ok = ~np.isnan(y)
                if (y[ok] == 1).sum() >= MIN_RECAL_POSITIVES and (y[ok] == 0).sum() > 0:
                    from sklearn.linear_model import LogisticRegression

                    x = logit(hist.loc[ok, f"p_{h}"].to_numpy()).reshape(-1, 1)
                    lr = LogisticRegression(C=1e4, max_iter=1000).fit(x, y[ok])
                    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
                    p_new[h][idx] = sigmoid(a * logit(out.p[h][idx]) + b)
                    maps.setdefault((model, str(m)), {})[h] = (a, b)
                    applied = True
        rows.append({"model": model, "month": str(m), "n": len(idx),
                     "drift_level": level, "drift_level_all_families": rep.level,
                     "max_psi": max(rep.psi_by_family.values(), default=0.0),
                     "recalibrated": applied, "recal_pool_n": len(hist)})
    p_new = monotone_in_horizon(p_new)
    return SystemOutput(p_new, out.uncertainty, out.disagreement), pd.DataFrame(rows), maps


def _apply_maps(df: pd.DataFrame, out: SystemOutput, maps: dict) -> SystemOutput:
    """Apply recalibration maps fitted on the main evaluation set to another table."""
    month = pd.to_datetime(df["as_of"]).dt.to_period("M").astype(str)
    p_new = {h: v.copy() for h, v in out.p.items()}
    for (model, m), idx in df.groupby([df["model"].astype(str), month]).indices.items():
        for h, (a, b) in maps.get((model, m), {}).items():
            p_new[h][idx] = sigmoid(a * logit(out.p[h][idx]) + b)
    return SystemOutput(monotone_in_horizon(p_new), out.uncertainty, out.disagreement)


def _weighted_ap(y_sorted: np.ndarray, w_sorted: np.ndarray) -> float:
    tp = np.cumsum(w_sorted * y_sorted)
    fp = np.cumsum(w_sorted * (1 - y_sorted))
    pos = tp[-1]
    if pos <= 0:
        return float("nan")
    prec = tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(prec * w_sorted * y_sorted) / pos)


class _Bootstrap:
    """Shared cluster-bootstrap weights (by serial) for paired comparisons."""

    def __init__(self, serials: np.ndarray, n_boot: int, seed: int = 0) -> None:
        codes = pd.factorize(serials)[0]
        rng = np.random.default_rng(seed)
        g = codes.max() + 1
        self.weights = [np.bincount(rng.integers(0, g, g), minlength=g)[codes]
                        for _ in range(n_boot)]

    def auprc(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        m = ~np.isnan(y)
        order = np.argsort(-p[m], kind="stable")
        ys = y[m][order]
        return np.array([_weighted_ap(ys, w[m][order].astype(float)) for w in self.weights])

    def mean(self, x: np.ndarray) -> np.ndarray:
        m = ~np.isnan(x)
        return np.array([np.sum(w[m] * x[m]) / max(w[m].sum(), 1) for w in self.weights])


# ------------------------------------------------------------------- runner


class ExperimentRunner:
    def __init__(self, spec: ExperimentSpec, overrides: dict | None = None,
                 cfg: AppConfig | None = None) -> None:
        self.spec = spec
        self.overrides = overrides or {}
        self.cfg = cfg or load_config(self.overrides or None)
        self.store = Store(self.cfg)
        reg = system_registry(self.cfg)
        unknown = [s for s in spec.systems if s not in reg]
        if unknown:
            raise ValueError(f"unknown systems {unknown}; known: {sorted(reg)}")
        self.systems = {s: reg[s] for s in spec.systems}
        idx = index_dir(self.cfg) / "cases"
        self.cases = CaseMemory.load(idx) if (idx / "indexes.json").exists() else None

    # ------------------------------------------------------------ main entry

    def run(self) -> dict[str, Any]:
        cfg, spec = self.cfg, self.spec
        started = time.perf_counter()
        tables = ["val", *spec.sets, *[EVAL_SETS[s] for s in spec.sets]]
        per_seed: list[dict] = []
        boot_store: dict = {}
        drift_tables: dict[str, pd.DataFrame] = {}
        selective: dict[str, pd.DataFrame] = {}
        reliability: dict[str, pd.DataFrame] = {}
        pr_curves: dict[str, tuple] = {}
        predictions: dict[str, dict[str, np.ndarray]] = {}
        for seed in cfg.seeds:
            bundle = load_bundle(cfg, seed)
            if bundle is None:
                raise RuntimeError(f"no trained bundle for seed {seed} in {bundle_dir(cfg, seed)}")
            ctx = EvalContext(cfg, self.store, bundle, self.cases, tables)
            seed_rows = []
            for set_name in spec.sets:
                df = ctx.df(set_name)
                if df.empty:
                    continue
                labels = _labels(df, cfg.horizons)
                if seed == cfg.seeds[0] and spec.bootstrap:
                    boot_store[set_name] = {"boot": _Bootstrap(
                        df["serial_number"].to_numpy(), cfg.evaluation.bootstrap_samples, 0),
                        "systems": {}}
                for name, sysdef in self.systems.items():
                    log.info("seed %d / %s / %s", seed, set_name, name)
                    out = sysdef.score(ctx, set_name)
                    lead_out = sysdef.score(ctx, EVAL_SETS[set_name])
                    if sysdef.drift_recalibration and cfg.features.use_drift:
                        out, dtab, maps = _drift_recalibrate(ctx, sysdef, set_name, out)
                        lead_out = _apply_maps(ctx.df(EVAL_SETS[set_name]), lead_out, maps)
                        if seed == cfg.seeds[0]:
                            drift_tables[f"{set_name}"] = dtab
                    row = self._metrics(ctx, sysdef, set_name, out, lead_out, labels)
                    row.update({"seed": seed, "set": set_name, "system": name})
                    seed_rows.append(row)
                    if seed == cfg.seeds[0]:
                        self._seed0_extras(ctx, sysdef, set_name, out, labels, boot_store,
                                           selective, reliability, pr_curves)
                        if spec.export_predictions:
                            self._keep_predictions(predictions.setdefault(set_name, {}),
                                                   df, labels, sysdef, out)
            per_seed.extend(seed_rows)
        results = pd.DataFrame(per_seed)
        summary = self._summarize(results, boot_store)
        return {"results": results, "summary": summary, "llm": None, "drift": drift_tables,
                "selective": selective, "reliability": reliability, "pr": pr_curves,
                "predictions": predictions, "elapsed_s": time.perf_counter() - started}

    def run_llm(self, result: dict[str, Any]) -> dict[str, Any]:
        """Add the graph runs on the stratified sample to a numeric result."""
        started = time.perf_counter()
        if self.spec.run_llm_sample:
            result["llm"] = self._llm_sample()
        result["elapsed_s"] += time.perf_counter() - started
        return result

    # --------------------------------------------------------------- metrics

    def _decisions(self, sysdef: SystemDef, out: SystemOutput, ratio: float,
                   df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        if not sysdef.cost_policy:
            cfg = cfg.model_copy(update={"features": cfg.features.model_copy(
                update={"use_cost_policy": False})})
        final, base, _ = decide_batch(cfg, out.p, ratio, out.uncertainty, out.disagreement)
        return final, base

    def _metrics(self, ctx: EvalContext, sysdef: SystemDef, set_name: str, out: SystemOutput,
                 lead_out: SystemOutput, labels: dict[int, np.ndarray]) -> dict:
        cfg = self.cfg
        row: dict[str, Any] = {"n_windows": len(next(iter(labels.values())))}
        for h in cfg.horizons:
            y, p = labels[h], out.p[h]
            row[f"positives_{h}"] = int(np.nansum(y))
            row[f"auprc_{h}"] = M.auprc(y, p)
            row[f"roc_auc_{h}"] = M.roc_auc(y, p)
            row[f"ece_{h}"] = M.ece(y, p, cfg.evaluation.ece_bins)
            for far in cfg.evaluation.far_levels:
                row[f"recall@far{far:g}_{h}"] = M.recall_at_far(y, p, far)
        h = cfg.max_horizon
        thr = M.threshold_at_far(labels[h], out.p[h], cfg.evaluation.far_levels[0])
        lt = M.lead_time(ctx.df(EVAL_SETS[set_name]), lead_out.p[h], thr)
        row.update({f"lead_{k}": v for k, v in lt.items()})
        df = ctx.df(set_name)
        for ratio in cfg.decision.cost_ratios:
            final, base = self._decisions(sysdef, out, ratio, df)
            for outcome in ("model", "oracle"):
                c = realized_costs(final, labels, ratio, cfg.actions, base, outcome)
                row[f"cost_{outcome}_r{ratio:g}"] = float(np.nanmean(c))
            no_esc = realized_costs(base, labels, ratio, cfg.actions)
            row[f"cost_noescalation_r{ratio:g}"] = float(np.nanmean(no_esc))
            row[f"escalation_rate_r{ratio:g}"] = float((final == "escalate_to_human").mean())
            hindsight = np.nanmin(np.column_stack([
                realized_costs(np.full(len(base), a, dtype=object), labels, ratio, cfg.actions)
                for a, s in cfg.actions.items() if not s.delegates]), axis=1)
            row[f"cost_hindsight_r{ratio:g}"] = float(np.nanmean(hindsight))
            monitor = realized_costs(np.full(len(base), "monitor", dtype=object), labels,
                                     ratio, cfg.actions)
            row[f"cost_always_monitor_r{ratio:g}"] = float(np.nanmean(monitor))
        return row

    def _seed0_extras(self, ctx, sysdef, set_name, out, labels, boot_store, selective,
                      reliability, pr_curves) -> None:
        cfg = self.cfg
        h = cfg.max_horizon
        key = f"{set_name}:{sysdef.name}"
        reliability[key] = M.reliability(labels[h], out.p[h], cfg.evaluation.ece_bins)
        y, p = labels[h], out.p[h]
        m = ~np.isnan(y)
        from sklearn.metrics import precision_recall_curve

        if len(np.unique(y[m])) == 2:
            prec, rec, _ = precision_recall_curve(y[m], p[m])
            step = max(1, len(prec) // 400)
            pr_curves[key] = (rec[::step], prec[::step])
        ratio = cfg.decision.default_cost_ratio
        final, base = self._decisions(sysdef, out, ratio, ctx.df(set_name))
        cost = realized_costs(base, labels, ratio, cfg.actions)
        hindsight = np.nanmin(np.column_stack([
            realized_costs(np.full(len(base), a, dtype=object), labels, ratio, cfg.actions)
            for a, s in cfg.actions.items() if not s.delegates]), axis=1)
        correct = np.where(np.isnan(cost), np.nan, (cost <= hindsight + 1e-9).astype(float))
        sel = M.selective_curve(correct, 1 - out.uncertainty, cfg.evaluation.coverage_levels)
        selective[key] = sel
        if set_name in boot_store:
            b = boot_store[set_name]["boot"]
            boot_store[set_name]["systems"][sysdef.name] = {
                "auprc": b.auprc(labels[h], out.p[h]),
                "cost": b.mean(realized_costs(final, labels, ratio, cfg.actions, base,
                                              "model")),
            }

    def _keep_predictions(self, cols: dict[str, np.ndarray], df: pd.DataFrame,
                          labels: dict[int, np.ndarray], sysdef: SystemDef,
                          out: SystemOutput) -> None:
        """Seed-0 per-window predictions (after drift recalibration) for one system."""
        cfg = self.cfg
        if not cols:
            cols["drive"] = pd.util.hash_pandas_object(
                df["serial_number"].astype(str), index=False).to_numpy()
            cols["model"] = df["model"].astype(str).to_numpy()
            cols["as_of"] = pd.to_datetime(df["as_of"]).to_numpy()
            for h in cfg.horizons:
                cols[f"y_{h}"] = labels[h].astype("float32")
        name = sysdef.name
        horizons = cfg.horizons if name in EXPORT_ALL_HORIZONS else [cfg.max_horizon]
        for h in horizons:
            cols[f"p{h}__{name}"] = np.asarray(out.p[h], dtype="float32")
        if name == "full_system":
            final, base = self._decisions(sysdef, out, cfg.decision.default_cost_ratio, df)
            cols["action__full_system"] = final.astype(str)
            cols["base_action__full_system"] = base.astype(str)

    def _summarize(self, results: pd.DataFrame, boot_store: dict) -> dict[str, pd.DataFrame]:
        cfg = self.cfg
        h = cfg.max_horizon
        metric_cols = [c for c in results.columns
                       if c not in ("seed", "set", "system") and results[c].dtype != object]
        agg = results.groupby(["set", "system"], sort=False)[metric_cols].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        summary = {"aggregate": agg.reset_index()}
        ci_rows = []
        for set_name, store in boot_store.items():
            systems = store["systems"]
            means = (results[(results["set"] == set_name) & (results["seed"] == cfg.seeds[0])]
                     .set_index("system"))
            baselines = [s for s in BASELINE_ORDER if s in systems]
            # reference = best baseline by AUPRC; ablation-only experiments use full_system
            if baselines:
                best = max(baselines, key=lambda s: means.loc[s, f"auprc_{h}"])
            else:
                best = "full_system" if "full_system" in systems else None
            for name, boot in systems.items():
                lo, hi = M.ci95(boot["auprc"])
                clo, chi = M.ci95(boot["cost"])
                row = {"set": set_name, "system": name,
                       f"auprc_{h}": means.loc[name, f"auprc_{h}"],
                       "auprc_ci_low": lo, "auprc_ci_high": hi,
                       f"cost_r{cfg.decision.default_cost_ratio:g}":
                           means.loc[name, f"cost_model_r{cfg.decision.default_cost_ratio:g}"],
                       "cost_ci_low": clo, "cost_ci_high": chi, "reference": best}
                if best and name != best:
                    d = boot["auprc"] - systems[best]["auprc"]
                    row["auprc_diff_vs_reference"] = float(np.nanmean(d))
                    row["auprc_p_value"] = M.paired_bootstrap_pvalue(d)
                    dc = boot["cost"] - systems[best]["cost"]
                    row["cost_diff_vs_reference"] = float(np.nanmean(dc))
                    row["cost_p_value"] = M.paired_bootstrap_pvalue(dc)
                ci_rows.append(row)
        summary["significance"] = pd.DataFrame(ci_rows)
        return summary

    # -------------------------------------------------------------- LLM sample

    def _llm_sample(self) -> dict[str, Any] | None:
        from pdm_agents.graph.builder import PredictionEngine, load_resources
        from pdm_agents.llm.client import create_llm
        from pdm_agents.llm.health import check_health
        from pdm_agents.tools.registry import build_tools

        cfg, spec = self.cfg, self.spec
        names = spec.llm_systems or [s for s, d in self.systems.items() if d.graph is not None]
        if not names:
            return None
        sample = stratified_sample(load_split(cfg, "test"), cfg)
        mode = spec.llm_mode
        note = ""
        if mode == "live":
            health = check_health(cfg.llm)
            if not health.reachable and spec.require_live:
                raise RuntimeError(f"live LLM unreachable ({health.error}) and the experiment "
                                   "requires live graph runs; set LLM_BASE_URL / LLM_API_KEY "
                                   "in server/.env to the running Colab server")
            if not health.reachable:
                note = (f"live LLM unreachable ({health.error}); graph runs used OFFLINE mode "
                        "(numeric pipeline + template explanations). These are not LLM results.")
                log.warning(note)
                mode = "offline"
        rows = []
        # Load data, models, indexes and the drift cache once; per system only the
        # config-dependent pieces (agents, feature flags, tools) change.
        shared = load_resources(cfg)
        # Offline runs are bound by local DuckDB work, so threads do not help there.
        workers = cfg.evaluation.llm_concurrency if mode != "offline" else 1
        for name in names:
            sysdef = self.systems[name]
            sys_cfg = load_config(deep_merge(self.overrides, sysdef.graph or {}))
            res = dataclasses.replace(shared, cfg=sys_cfg, tools=build_tools(
                sys_cfg, shared.bundle, shared.cases, shared.knowledge))
            engine = PredictionEngine(sys_cfg, resources=res,
                                      llm=create_llm(sys_cfg.llm, mode) if mode != "offline"
                                      else None, checkpoint=False, record_runs=False)
            requests = [PredictRequest(serial_number=r.serial_number,
                                       as_of_date=pd.Timestamp(r.as_of).date(),
                                       horizons=cfg.horizons, offline=mode == "offline")
                        for r in sample.itertuples()]
            log.info("graph runs: %s on %d windows (%s)", name, len(requests), mode)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                responses = list(pool.map(engine.predict, requests))
            rows.append(self._llm_metrics(name, sample, responses))
        table = pd.DataFrame(rows)
        return {"table": table, "mode": mode, "note": note, "n": len(sample),
                "positives": int((sample[label_column(cfg.max_horizon)] == 1).sum())}

    def _llm_metrics(self, name: str, sample: pd.DataFrame, responses: list) -> dict:
        cfg = self.cfg
        h = cfg.max_horizon
        y = sample[label_column(h)].to_numpy("float64")
        p = np.array([r.fused.probabilities.get(h, np.nan) if r.fused else np.nan
                      for r in responses])
        exp = [r.explanation for r in responses if r.explanation]
        grounded = [_numeric_grounding(r) for r in responses if r.explanation]
        grounded = [g for g in grounded if g is not None]
        return {
            "system": name,
            "n": len(responses),
            f"auprc_{h}_sample": M.auprc(y, p),
            "llm_available_share": float(np.mean([r.llm_available for r in responses])),
            "mean_llm_calls": float(np.mean([r.llm_calls for r in responses])),
            "mean_latency_s": float(np.mean([r.latency_ms for r in responses]) / 1000),
            "p95_latency_s": float(np.percentile([r.latency_ms for r in responses], 95) / 1000),
            "escalation_share": float(np.mean([bool(r.decision and r.decision.escalated)
                                               for r in responses])),
            "llm_explanation_share": float(np.mean([e.source == "llm" for e in exp]))
            if exp else float("nan"),
            "explanations_with_valid_citation": float(np.mean([
                bool(e.cited_finding_ids or e.cited_case_ids or e.cited_knowledge_ids)
                for e in exp])) if exp else float("nan"),
            "mean_citations": float(np.mean([len(e.cited_finding_ids) + len(e.cited_case_ids)
                                             + len(e.cited_knowledge_ids) for e in exp]))
            if exp else float("nan"),
            "numbers_grounded_share": float(np.mean(grounded)) if grounded else float("nan"),
            "agent_failed_share": float(np.mean([any(f.status == "failed" for f in r.findings)
                                                 for r in responses])),
            "replanned_share": float(np.mean([r.context_summary.get("attempts", 1) > 1
                                              for r in responses])),
            "planner_llm_share": float(np.mean([any(
                e.node == "planner" and e.data.get("source") == "llm" for e in r.trace)
                for r in responses])),
            "mean_agents_run": float(np.mean([len(r.plan) for r in responses])),
        }


NUM_RE = re.compile(r"(?<![\w-])(\d+(?:\.\d+)?)(%?)")


def _numeric_grounding(resp) -> float | None:
    """Share of numbers in the explanation that match a number in the evidence it was given
    (faithfulness proxy). Percentages are compared on the 0-100 scale."""
    text = resp.explanation.text
    nums = [(float(v), pct == "%") for v, pct in NUM_RE.findall(re.sub(r"\[[^\]]*\]", "", text))]
    if not nums:
        return None
    facts_text = resp.model_dump_json(exclude={"explanation", "trace", "request", "run_id"})
    facts = [float(v) for v, _ in NUM_RE.findall(facts_text)]
    facts += [f * 100 for f in facts if f <= 1]

    def grounded(x: float) -> bool:
        return any(abs(x - f) <= max(0.051, 0.01 * abs(f)) for f in facts)

    return float(np.mean([grounded(x) for x, _ in nums]))


def stratified_sample(test: pd.DataFrame, cfg: AppConfig, seed: int = 0) -> pd.DataFrame:
    """All positives (up to the cap) + random negatives, from the test set."""
    ymax = label_column(cfg.max_horizon)
    rng = np.random.default_rng(seed)
    pos = test[test[ymax] == 1]
    neg = test[test[ymax] == 0]
    n_pos = min(len(pos), cfg.evaluation.llm_sample_max_positives,
                cfg.evaluation.llm_sample_size)
    pos = pos.iloc[rng.choice(len(pos), n_pos, replace=False)] if n_pos < len(pos) else pos
    n_neg = min(len(neg), cfg.evaluation.llm_sample_size - len(pos))
    neg = neg.iloc[rng.choice(len(neg), n_neg, replace=False)]
    return pd.concat([pos, neg]).reset_index(drop=True)


def run_experiment(path: Path, out_root: Path | None = None) -> Path:
    """Run the experiment YAML at ``path``; returns the report directory."""
    from pdm_agents.evaluation.report import write_report

    spec, overrides = ExperimentSpec.from_yaml(path)
    runner = ExperimentRunner(spec, overrides)
    result = runner.run()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = (out_root or runner.cfg.paths.reports) / f"{spec.name}-{stamp}"
    write_report(out, spec, runner, result)  # numeric results are safe on disk first
    if spec.run_llm_sample:
        log.info("numeric report written; starting graph runs on the stratified sample")
        write_report(out, spec, runner, runner.run_llm(result))
    return out
