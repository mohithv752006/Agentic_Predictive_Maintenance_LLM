"""Shared fixtures: a synthetic dataset prepared once per test session."""

from __future__ import annotations

import pytest

from pdm_agents.config import AppConfig, ensure_dirs, load_config
from tests.synthetic import build_synthetic_zips, synthetic_overrides


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """(root dir, ground truth) of the synthetic raw zips."""
    root = tmp_path_factory.mktemp("synth")
    truth = build_synthetic_zips(root / "raw")
    return root, truth


@pytest.fixture(scope="session")
def synth_cfg(synth) -> AppConfig:
    root, _ = synth
    cfg = load_config(synthetic_overrides(root), env_file=root / "no.env")
    ensure_dirs(cfg)
    return cfg


@pytest.fixture(scope="session")
def prepared_cfg(synth_cfg) -> AppConfig:
    """Synthetic data harmonized into Parquet."""
    from pdm_agents.data.harmonize import prepare

    prepare(synth_cfg, workers=2)
    return synth_cfg


@pytest.fixture(scope="session")
def featured_cfg(prepared_cfg) -> AppConfig:
    """Synthetic data with labels, splits and feature tables built."""
    from pdm_agents.data.features import build_feature_tables
    from pdm_agents.data.store import Store

    build_feature_tables(prepared_cfg, Store(prepared_cfg))
    return prepared_cfg


@pytest.fixture(scope="session")
def trained_cfg(featured_cfg) -> AppConfig:
    """Synthetic data with a trained model bundle saved for the first seed."""
    from pdm_agents.data.store import Store
    from pdm_agents.models.bundle import bundle_dir, train_bundle

    bundle = train_bundle(featured_cfg, Store(featured_cfg), seed=featured_cfg.seeds[0])
    bundle.save(bundle_dir(featured_cfg, featured_cfg.seeds[0]))
    return featured_cfg


@pytest.fixture(scope="session")
def indexed_cfg(trained_cfg) -> AppConfig:
    """Synthetic data with trained bundle, case memory and knowledge index."""
    from scripts.index import build_indexes

    build_indexes(trained_cfg)
    return trained_cfg


@pytest.fixture(scope="session")
def resources(indexed_cfg):
    from pdm_agents.graph.builder import load_resources

    return load_resources(indexed_cfg)


@pytest.fixture(scope="session")
def sample_serial(indexed_cfg) -> tuple[str, str]:
    """(serial, as_of) of a test-split window that fails within 30 days."""
    from pdm_agents.data.features import load_split

    test = load_split(indexed_cfg, "test")
    pos = test[test["y_30"] == 1]
    row = (pos if not pos.empty else test).iloc[0]
    return str(row["serial_number"]), str(row["as_of"])[:10]
