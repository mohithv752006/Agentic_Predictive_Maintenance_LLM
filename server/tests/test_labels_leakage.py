"""Leakage-safety and label correctness (spec section 13)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from pdm_agents.data.features import compute_features, feature_columns, load_split
from pdm_agents.data.labels import SPLITS, label_column, load_periods
from pdm_agents.data.store import Store


@pytest.fixture(scope="module")
def tables(featured_cfg):
    return {s: load_split(featured_cfg, s) for s in SPLITS}


def test_no_serial_overlap_across_splits(tables):
    sets = {s: set(t["serial_number"]) for s, t in tables.items() if not t.empty}
    assert {"train", "val", "test", "unseen"} <= set(sets)
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not sets[a] & sets[b], f"serial overlap between {a} and {b}"


def test_unseen_split_holds_only_holdout_models(featured_cfg, tables):
    meta = Store(featured_cfg).models_meta()
    assert set(tables["unseen"]["model"]) == set(meta["holdout_models"])
    for s in ("train", "val", "test"):
        assert not set(tables[s]["model"]) & set(meta["holdout_models"])


def test_temporal_periods_and_label_horizon_inside_period(featured_cfg, tables):
    periods = load_periods(Store(featured_cfg))
    h = featured_cfg.max_horizon
    for split in SPLITS:
        t = tables[split]
        start, end = periods.bounds(split)
        as_of = pd.to_datetime(t["as_of"]).dt.date
        assert as_of.min() >= start
        assert (as_of + timedelta(days=h)).max() <= end, f"{split} labels leak past period"
    assert periods.bounds("train")[1] < periods.bounds("val")[0] <= periods.bounds("val")[1]
    assert periods.bounds("val")[1] < periods.bounds("test")[0]


def test_labels_match_ground_truth(synth, tables):
    _, truth = synth
    checked = 0
    for t in tables.values():
        for row in t.itertuples():
            fail = truth[row.serial_number]["fail_date"]
            end = truth[row.serial_number]["end"]
            as_of = pd.Timestamp(row.as_of).date()
            assert fail is None or as_of < fail  # no windows on/after failure
            for h in (7, 14, 30):
                y = getattr(row, label_column(h))
                if fail is not None and 1 <= (fail - as_of).days <= h:
                    assert y == 1
                elif fail is None and (end - as_of).days < h:
                    assert np.isnan(y)  # censored: left the fleet without failing
                else:
                    assert y == 0
                checked += 1
    assert checked > 1000


def test_training_sampling_ratio_and_eval_natural(featured_cfg, tables):
    ymax = label_column(featured_cfg.max_horizon)
    train = tables["train"]
    n_pos, n_neg = int((train[ymax] == 1).sum()), int((train[ymax] == 0).sum())
    assert n_pos > 0
    assert n_neg <= featured_cfg.data.neg_pos_ratio * n_pos
    # evaluation splits are not rebalanced: positive rate far below the training rate
    for s in ("val", "test"):
        rate = (tables[s][ymax] == 1).mean()
        assert rate < (n_pos / (n_pos + n_neg))


def test_features_never_use_future_data(featured_cfg, tables):
    store = Store(featured_cfg)
    sample = tables["test"].groupby("model").head(3).reset_index(drop=True)
    cutoff = pd.to_datetime(sample["as_of"]).min().date()
    sample = sample[pd.to_datetime(sample["as_of"]).dt.date == cutoff]
    if sample.empty:
        sample = tables["test"].head(1)
        cutoff = pd.Timestamp(sample["as_of"].iloc[0]).date()
    keys = sample[["sample_id", "serial_number", "model", "as_of", "capacity_bytes"]]
    store.cursor().execute(
        f"CREATE OR REPLACE VIEW smart_cut AS SELECT * FROM smart WHERE date <= DATE '{cutoff}'"
    )
    full = compute_features(featured_cfg, store, keys).sort_values("sample_id")
    cut = compute_features(featured_cfg, store, keys, smart_view="smart_cut").sort_values(
        "sample_id")
    cols = feature_columns(full)
    pd.testing.assert_frame_equal(full[cols].reset_index(drop=True),
                                  cut[cols].reset_index(drop=True))


def test_feature_values_on_degrading_drive(featured_cfg, tables):
    train = tables["train"]
    pos = train[train[label_column(7)] == 1]
    neg = train[train[label_column(30)] == 0]
    # synthetic failures accumulate reallocated sectors right before failing
    assert pos["f__smart_5_raw__slope"].median() > neg["f__smart_5_raw__slope"].median()
    assert pos["f__smart_5_raw__n_inc"].median() > neg["f__smart_5_raw__n_inc"].median()
    assert (train["f__ctx__n_obs"] <= featured_cfg.data.window_days).all()
