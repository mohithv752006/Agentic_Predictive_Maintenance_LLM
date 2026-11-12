"""Models: calibration, fusion, bundle training/persistence on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from pdm_agents.data.features import load_split
from pdm_agents.data.labels import label_column
from pdm_agents.data.store import Store
from pdm_agents.models.bundle import load_bundle
from pdm_agents.models.calibration import Calibrator
from pdm_agents.models.fusion import Fusion, disagreement, uncertainty


def test_calibrators_map_scores_to_rates():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0, 0.2, 20_000)
    y = (rng.uniform(size=p_true.size) < p_true).astype(float)
    raw = np.sqrt(p_true)  # monotone distortion
    for method in ("isotonic", "platt"):
        cal = Calibrator(method).fit(raw, y)
        out = cal.transform(raw)
        assert abs(out.mean() - y.mean()) < 0.01
        again = Calibrator.from_dict(cal.to_dict()).transform(raw)
        np.testing.assert_allclose(out, again)
    assert np.allclose(Calibrator("identity").transform([0.2, 0.7]), [0.2, 0.7])


def test_fusion_methods_and_missing_agent():
    rng = np.random.default_rng(1)
    n = 5000
    y = (rng.uniform(size=n) < 0.05).astype(float)
    a = np.clip(0.03 + 0.5 * y + rng.normal(0, 0.05, n), 0.001, 0.999)
    b = np.clip(0.05 + rng.normal(0, 0.02, n), 0.001, 0.999)  # uninformative
    prior = np.full(n, 0.05)
    f = Fusion(["a", "b"], 30).fit({"a": a, "b": b}, prior, y)
    assert f.coef_[0] > f.coef_[1]
    for m in ("stacking", "log_odds", "majority_vote"):
        p = f.predict({"a": a, "b": b}, prior, m)
        assert p.shape == (n,) and np.all((p >= 0) & (p <= 1))
    only_b = f.predict({"b": b}, prior, "stacking")  # a missing -> imputed with prior
    assert np.isfinite(only_b).all()
    g = Fusion.from_dict(f.to_dict())
    np.testing.assert_allclose(g.predict({"a": a, "b": b}, prior), f.predict({"a": a, "b": b},
                                                                              prior))


def test_disagreement_and_uncertainty_ranges():
    assert disagreement({"a": 0.0, "b": 1.0}) == pytest.approx(1.0)
    assert disagreement({"a": 0.1}) == 0.0
    # an agent the fusion does not rely on cannot create disagreement
    assert disagreement({"a": 0.9, "b": 0.0}, {"a": 1.0, "b": 0.0}) == 0.0
    assert disagreement({"a": 0.9, "b": 0.1}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.8)
    assert uncertainty(0.5, 1.0) == pytest.approx(1.0)
    assert uncertainty(0.001, 1.0) < 0.05
    assert uncertainty(0.001, 0.5) > 0.5  # missing evidence inflates uncertainty


def test_drift_ignores_aging_of_cumulative_counters():
    from pdm_agents.decision.drift import drift_feature

    assert not drift_feature("f__smart_9_raw__last")  # power-on hours grow with age
    assert not drift_feature("f__smart_241_raw__max")
    assert drift_feature("f__smart_9_raw__slope")  # usage rate can genuinely drift
    assert drift_feature("f__smart_194_raw__last")  # temperature level is not cumulative
    assert drift_feature("f__smart_7_normalized__last")
    assert not drift_feature("f__ctx__n_obs")


@pytest.mark.slow
def test_bundle_trains_saves_loads_and_scores(trained_cfg):
    cfg = trained_cfg
    bundle = load_bundle(cfg)
    assert bundle is not None
    assert set(bundle.families) == {"media_health", "mechanical", "thermal", "usage_wear"}
    test = load_split(cfg, "test")
    risks = bundle.all_family_risks(test)
    fused = bundle.fuse(risks, test["model"].tolist())
    glob = bundle.global_risk(test)
    head = test.head(200)
    seq = bundle.sequence_risk(cfg, Store(cfg), head)
    for h in cfg.horizons:
        for p in (fused[h], glob[h], risks["media_health"][h]):
            assert p.shape == (len(test),) and np.all((p >= 0) & (p <= 1))
        assert seq[h].shape == (len(head),)
    # media health is the informative family on synthetic data
    y = test[label_column(30)].to_numpy()
    m = ~np.isnan(y)
    pos, neg = risks["media_health"][30][m][y[m] == 1], risks["media_health"][30][m][y[m] == 0]
    assert pos.mean() > neg.mean()
    # unseen model falls back to pooled prior
    prior = bundle.prior(["NOT A MODEL"])
    assert prior[30][0] == bundle.base_rates["__pooled__"][30]
    # calibrators and fusion were fit on validation data, not training data
    assert bundle.meta["n_val"] == len(load_split(cfg, "val"))
