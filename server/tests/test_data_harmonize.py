"""Harmonization across differing column sets, model selection, store queries."""

from __future__ import annotations

import zipfile
from datetime import date

import pandas as pd

from pdm_agents.data.harmonize import list_day_files, read_day_csv
from pdm_agents.data.store import Store, summarize


def _read_member(zip_path, member):
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(member)


def test_schema_harmonization_across_quarters(synth_cfg):
    files = list_day_files(sorted(synth_cfg.paths.raw.glob("*.zip")))
    assert all("__MACOSX" not in f.member for f in files)
    q1 = next(f for f in files if f.quarter == 1)
    q2 = next(f for f in files if f.quarter == 2)
    cols = synth_cfg.data.smart_columns
    t1 = read_day_csv(_read_member(q1.zip_path, q1.member), cols)
    t2 = read_day_csv(_read_member(q2.zip_path, q2.member), cols)
    assert t1.schema == t2.schema  # identical schema despite different CSV headers
    assert t1.column_names[:5] == ["date", "serial_number", "model", "capacity_bytes", "failure"]
    assert "smart_999_raw" not in t2.column_names  # unknown column dropped
    assert t1["smart_184_raw"].null_count == t1.num_rows  # absent in Q1 -> null
    assert t2["smart_184_raw"].null_count == 0
    assert str(t1.schema.field("smart_5_raw").type) == "double"


def test_prepare_selects_models_and_writes_partitions(prepared_cfg, synth):
    _, truth = synth
    store = Store(prepared_cfg)
    assert store.ready
    meta = store.models_meta()
    assert len(meta["training_models"]) == 3
    assert len(meta["holdout_models"]) == 1
    assert "SYN E5" not in meta["training_models"] + meta["holdout_models"]  # no failures
    parts = list((prepared_cfg.paths.parquet / "smart").glob("model=*/year=*/quarter=*"))
    assert len(parts) == 4 * 2
    drives = store.df("SELECT * FROM drives")
    for row in drives.itertuples():
        expected = truth[row.serial_number]["fail_date"]
        got = None if pd.isna(row.failure_date) else pd.Timestamp(row.failure_date).date()
        assert got == expected
    assert drives["holdout_model"].sum() == (drives["model"] == meta["holdout_models"][0]).sum()
    assert "SYN" in summarize(store)


def test_duplicates_removed_and_window_respects_as_of(prepared_cfg):
    store = Store(prepared_cfg)
    dup = store.df(
        "SELECT serial_number, date, count(*) n FROM smart GROUP BY ALL HAVING count(*) > 1"
    )
    assert dup.empty
    serial = store.df("SELECT serial_number FROM drives WHERE failure_date IS NOT NULL LIMIT 1")
    s = serial.iloc[0, 0]
    as_of = date(2023, 3, 15)
    w = store.drive_window(s, as_of, 30)
    assert not w.empty
    assert pd.to_datetime(w["date"]).max().date() <= as_of
    assert pd.to_datetime(w["date"]).min().date() >= date(2023, 2, 14)
