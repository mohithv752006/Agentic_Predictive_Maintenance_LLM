"""Record the data for the public demo dashboard (client built with VITE_DEMO=1).

Talks to a running API server (python main.py) and writes JSON + report plots to
client/public/demo/. Only runs made with the live LLM are exported.

Usage:
  python -m scripts.export_demo --analyze     # first run new analyses (needs the Colab LLM)
  python -m scripts.export_demo               # export what is already recorded
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import typer

from pdm_agents.config import AppConfig
from pdm_agents.data.store import Store
from scripts._common import bootstrap

app = typer.Typer(add_completion=False)
CLIENT_DEMO = Path(__file__).resolve().parents[2] / "client" / "public" / "demo"
TEST_START = pd.Timestamp("2025-08-08")  # first day of the test period (splits.json val_end + 1)


def select_drives(cfg: AppConfig, per_model: int, healthy_per_model: int) -> list[tuple[str, str]]:
    """Failing drives (as-of a few days before failure) and healthy drives from test/unseen."""
    store = Store(cfg)
    meta = store.meta_dir
    with (meta / "drives.parquet").open("rb") as fh:
        drives = pd.read_parquet(fh)
    with (meta / "drive_splits.parquet").open("rb") as fh:
        splits = pd.read_parquet(fh)
    df = drives.merge(splits[["serial_number", "split"]], on="serial_number")
    df = df[df["split"].isin(["test", "unseen"])].sort_values("serial_number")
    out: list[tuple[str, str]] = []
    offsets = [3, 10, 20, 6]
    for _, g in df.groupby("model"):
        failed_at = pd.to_datetime(g["failure_date"])
        fail = g[failed_at.notna() & (failed_at >= TEST_START + pd.Timedelta(days=45))]
        for i, r in enumerate(fail.head(per_model).itertuples()):
            as_of = pd.Timestamp(r.failure_date) - pd.Timedelta(days=offsets[i % 4])
            out.append((r.serial_number, str(as_of.date())))
        ended = pd.to_datetime(g["last_date"]) >= pd.Timestamp("2025-12-31")
        ok = g[failed_at.isna() & ended]
        out += [(r.serial_number, "2025-12-20") for r in ok.head(healthy_per_model).itertuples()]
    return out


def analyze(client: httpx.Client, pairs: list[tuple[str, str]], cost_ratio: float,
            workers: int) -> None:
    def one(pair: tuple[str, str]) -> str:
        serial, as_of = pair
        r = client.post("/predict", json={"serial_number": serial, "as_of_date": as_of,
                                          "cost_ratio": cost_ratio}, timeout=900)
        if r.status_code != 200:
            return f"{serial} {as_of}: HTTP {r.status_code} {r.text[:120]}"
        body = r.json()
        src = (body.get("explanation") or {}).get("source")
        return (f"{serial} {as_of}: {body['decision']['action'] if body.get('decision') else '-'}"
                f" (llm={body['llm_available']}, explanation={src})")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(one, pairs):
            typer.echo("  " + line)


def live_runs(cfg: AppConfig) -> list[dict]:
    """Latest live-LLM run per (serial, as-of) from the server's run store."""
    con = sqlite3.connect(str(cfg.paths.runtime / "runs.sqlite"))
    rows = con.execute("SELECT created_at, response FROM runs ORDER BY created_at DESC").fetchall()
    seen, out = set(), []
    for _, resp in rows:
        r = json.loads(resp)
        exp = r.get("explanation") or {}
        if not (r.get("llm_available") and exp.get("source") == "llm"):
            continue
        key = (r["request"]["serial_number"], r["context_summary"].get("as_of_date")
               or r["request"].get("as_of_date"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


@app.command()
def main(
    server: str = typer.Option("http://127.0.0.1:8000", help="running API server"),
    do_analyze: bool = typer.Option(False, "--analyze", help="run new live analyses first"),
    per_model: int = typer.Option(3, help="failing drives per drive model"),
    healthy_per_model: int = typer.Option(1, help="healthy drives per drive model"),
    cost_ratio: float = typer.Option(50.0),
    workers: int = typer.Option(4),
    out: Path = typer.Option(CLIENT_DEMO, help="output folder (replaced)"),
) -> None:
    cfg = bootstrap()
    client = httpx.Client(base_url=server, timeout=120)
    health = client.get("/health").json()
    if do_analyze:
        if not health["llm"]["reachable"]:
            raise typer.BadParameter("the server's LLM is not reachable; start Colab and save the "
                                     "connection on the dashboard Settings page first")
        pairs = select_drives(cfg, per_model, healthy_per_model)
        typer.echo(f"analysing {len(pairs)} drives with the live LLM ({workers} at a time)...")
        analyze(client, pairs, cost_ratio, workers)

    runs = live_runs(cfg)
    if not runs:
        raise typer.BadParameter("no live-LLM runs recorded; use --analyze with the LLM running")
    if out.exists():
        if any(out.iterdir()) and not (out / "meta.json").exists():
            raise typer.BadParameter(f"{out} is not a demo export; refusing to replace it")
        shutil.rmtree(out)

    # secrets that must never reach the public site
    saved = cfg.paths.runtime / "settings.json"
    secrets = {cfg.llm.base_url, cfg.llm.api_key}
    if saved.exists():
        s = json.loads(saved.read_text(encoding="utf-8"))
        secrets |= {s.get("llm_base_url", ""), s.get("llm_api_key", "")}
    secrets = {x.rstrip("/").removesuffix("/v1") for x in secrets if x and len(x) > 8}

    index: dict[str, list[dict]] = {}
    for r in runs:
        serial = r["request"]["serial_number"]
        as_of = str(r["context_summary"].get("as_of_date") or r["request"].get("as_of_date"))
        ratio = r["request"].get("cost_ratio") or cost_ratio
        index.setdefault(serial, []).append({"as_of": as_of, "run_id": r["run_id"],
                                             "cost_ratio": ratio})
        _write(out / "runs" / f"{r['run_id']}.json", r)
    _write(out / "analyses" / "index.json", index)

    latest_action = {r["request"]["serial_number"]: (r.get("decision") or {}).get("action")
                     for r in reversed(runs)}
    items = []
    for serial in index:
        history = client.get(f"/drives/{serial}").raise_for_status().json()
        _write(out / "drives" / f"{serial}.json", history)
        listing = client.get("/drives", params={"search": serial, "page_size": 20}).json()
        item = next(i for i in listing["items"] if i["serial_number"] == serial)
        item["action"] = latest_action.get(serial) or item["action"]
        items.append(item)
    _write(out / "drives.json", {"items": items, "models": listing["models"],
                                 "risk_horizon": listing["risk_horizon"]})
    _write(out / "fleet_stats.json", client.get("/drives/stats").json())

    health["llm"].update(base_url="", error=None, reachable=True)
    _write(out / "health.json", health)
    settings = client.get("/settings").json()
    settings.update(llm_base_url="", llm_api_key_set=False, llm_api_key_hint="")
    _write(out / "settings.json", settings)
    run_ids = {r["run_id"] for r in runs}
    esc = [dict(e, feedback=None) for e in client.get("/escalations").json()
           if e["run_id"] in run_ids]
    _write(out / "escalations.json", esc)

    experiments = client.get("/experiments").json()
    keep = []
    for prefix in ("final_results", "ablation_llm"):
        match = [e for e in experiments if e["id"].startswith(prefix + "-")]
        if match:
            keep.append(match[0])  # newest first
    _write(out / "experiments.json", keep)
    for e in keep:
        detail = client.get(f"/experiments/{e['id']}").json()
        detail["config"] = {k: detail["config"].get(k) for k in ("experiment", "overrides")}
        _write(out / "experiments" / f"{e['id']}.json", detail)
        for url in detail["plots"]:
            dest = out / url.lstrip("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(client.get(url).raise_for_status().content)

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    _write(out / "meta.json", {"generated_at": stamp,
                               "drives": len(index), "runs": len(runs),
                               "experiments": [e["id"] for e in keep]})

    leaks = [p for p in out.rglob("*.json")
             if any(s in p.read_text(encoding="utf-8") for s in secrets)]
    if leaks:
        shutil.rmtree(out)
        raise typer.BadParameter(f"secret found in {leaks[:3]}; export deleted")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    typer.echo(f"demo written to {out}: {len(index)} drives, {len(runs)} runs, "
               f"{len(keep)} experiments, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    app()
