"""Evaluation: metric correctness and an end-to-end experiment on synthetic data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from pdm_agents.evaluation import metrics as M
from pdm_agents.evaluation.runner import (
    ExperimentRunner,
    ExperimentSpec,
    _weighted_ap,
    stratified_sample,
    system_registry,
)
from tests.synthetic import synthetic_overrides


def test_recall_at_far_and_threshold():
    y = np.array([0] * 100 + [1] * 10, dtype=float)
    p = np.concatenate([np.linspace(0, 0.5, 100), np.linspace(0.45, 0.95, 10)])
    thr = M.threshold_at_far(y, p, 0.01)
    assert (p[:100] >= thr).mean() <= 0.01
    assert M.recall_at_far(y, p, 0.01) == pytest.approx((p[100:] >= thr).mean())


def test_ece_perfect_and_bad():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.2, 50_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert M.ece(y, p) < 0.01
    assert M.ece(y, np.clip(p * 3, 0, 1)) > 0.05


def test_weighted_ap_matches_sklearn_with_unit_weights():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p * 0.3).astype(float)
    order = np.argsort(-p)
    assert _weighted_ap(y[order], np.ones(2000)) == pytest.approx(
        average_precision_score(y, p), abs=1e-9)


def test_lead_time():
    df = pd.DataFrame({"serial_number": ["a", "a", "a", "b", "b"],
                       "days_to_failure": [10, 5, 1, 8, 2]})
    out = M.lead_time(df, np.array([0.1, 0.9, 0.9, 0.0, 0.0]), 0.5)
    assert out["detected_share"] == 0.5 and out["mean_lead_days"] == 5


def test_selective_curve_monotone_coverage():
    correct = np.array([1, 1, 0, 1, 0, 0], dtype=float)
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    sel = M.selective_curve(correct, conf, [0.5, 1.0])
    assert sel.iloc[0]["selective_accuracy"] == pytest.approx(2 / 3)
    assert sel.iloc[1]["selective_accuracy"] == pytest.approx(0.5)


def test_every_system_and_ablation_registered(indexed_cfg):
    reg = system_registry(indexed_cfg)
    for name in ["lgbm_global", "sequence", "single_llm_agent", "multi_agent_no_rag",
                 "full_system", "no_case_retrieval", "no_knowledge_retrieval",
                 "no_calibration", "majority_vote", "no_cost_policy",
                 "no_drift_recalibration", "no_planner",
                 *[f"no_{a}" for a in indexed_cfg.numeric_agents()]]:
        assert name in reg
    assert reg["no_planner"].graph == {"features": {"use_planner": False}}


def test_require_live_fails_instead_of_running_offline(indexed_cfg, monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    overrides = {**synthetic_overrides(indexed_cfg.paths.raw.parent), "llm": {"base_url": ""}}
    spec = ExperimentSpec(name="live_only", systems=["full_system"], sets=["test"],
                          llm_mode="live", require_live=True, llm_systems=["full_system"])
    runner = ExperimentRunner(spec, overrides)
    with pytest.raises(RuntimeError, match="requires live graph runs"):
        runner._llm_sample()


@pytest.mark.slow
def test_experiment_end_to_end(indexed_cfg, tmp_path, monkeypatch):
    from pdm_agents.evaluation.report import write_report

    monkeypatch.setenv("LLM_API_KEY", "sk-secret-never-in-reports")
    overrides = synthetic_overrides(indexed_cfg.paths.raw.parent)
    reg = system_registry(indexed_cfg)
    spec = ExperimentSpec(name="synthetic", systems=list(reg), sets=["test", "unseen"],
                          llm_mode="fake", run_llm_sample=True,
                          llm_systems=["full_system", "single_llm_agent"],
                          export_predictions=True)
    runner = ExperimentRunner(spec, overrides)
    result = runner.run_llm(runner.run())
    res = result["results"]
    assert set(res["system"]) == set(reg)
    assert set(res["set"]) == {"test", "unseen"}
    assert res["auprc_30"].notna().any()
    # numeric results never come from the LLM: identical by construction
    t = res[res["set"] == "test"].set_index("system")
    assert t.loc["single_llm_agent", "auprc_30"] == t.loc["lgbm_global", "auprc_30"]
    assert t.loc["no_planner", "auprc_30"] == t.loc["full_system", "auprc_30"]
    llm = result["llm"]
    assert llm["mode"] == "fake" and set(llm["table"]["system"]) == {"full_system",
                                                                      "single_llm_agent"}
    assert (llm["table"]["mean_llm_calls"] > 0).all()
    out = tmp_path / "report"
    write_report(out, spec, runner, result)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["name"] == "synthetic" and "full_system" in metrics["headline"]
    assert (out / "tables" / "prediction_test.csv").exists()
    assert (out / "tables" / "prediction_test.tex").exists()
    assert (out / "tables" / "cost_test.csv").exists()
    assert (out / "tables" / "llm_sample.csv").exists()
    assert list((out / "plots").glob("*.png"))
    for set_name in ("test", "unseen"):
        pred = pd.read_parquet(out / "predictions" / f"{set_name}.parquet")
        rows = res[(res["set"] == set_name) & (res["system"] == "full_system")]
        assert len(pred) == rows["n_windows"].iloc[0]
        assert {"drive", "model", "as_of", "y_7", "y_14", "y_30", "p30__full_system",
                "p7__full_system", "p30__no_planner", "action__full_system"} <= set(pred.columns)
        assert "p7__no_planner" not in pred.columns
        assert "serial_number" not in pred.columns
        assert pred["p30__full_system"].between(0, 1).all()
    for f in out.rglob("*"):
        if f.is_file() and f.suffix in (".yaml", ".json", ".csv", ".tex"):
            assert "sk-secret-never-in-reports" not in f.read_text(encoding="utf-8"), f


def test_stratified_sample_composition(indexed_cfg):
    from pdm_agents.data.features import load_split

    test = load_split(indexed_cfg, "test")
    s = stratified_sample(test, indexed_cfg)
    ymax = "y_30"
    assert len(s) <= indexed_cfg.evaluation.llm_sample_size
    assert (s[ymax] == 1).sum() == min((test[ymax] == 1).sum(),
                                       indexed_cfg.evaluation.llm_sample_max_positives)
