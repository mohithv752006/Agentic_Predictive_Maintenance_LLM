"""Run the full multi-agent prediction for one drive from the command line.

Usage:
  python -m scripts.predict --serial ZL2ABC12 [--as-of 2025-11-01] [--cost-ratio 50]
                            [--offline | --fake] [--json]

``--offline`` skips the LLM (numeric pipeline + template explanation);
``--fake`` uses the deterministic FakeLLM (useful without a Colab endpoint).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from pdm_agents.graph.builder import PredictionEngine
from pdm_agents.schemas import PredictRequest
from scripts._common import bootstrap

app = typer.Typer(add_completion=False)


@app.command()
def main(
    serial: str = typer.Option(..., help="Drive serial number"),
    as_of: str | None = typer.Option(None, help="As-of date YYYY-MM-DD (default: latest)"),
    cost_ratio: float | None = typer.Option(None, help="Missed-failure cost ratio"),
    offline: bool = typer.Option(False, help="Do not call the LLM"),
    fake: bool = typer.Option(False, help="Use the deterministic FakeLLM"),
    as_json: bool = typer.Option(False, "--json", help="Print the full PredictResponse"),
    config: Path | None = typer.Option(None, help="Extra YAML merged over default.yaml"),
) -> None:
    cfg = bootstrap(config_file=config)
    engine = PredictionEngine(cfg, llm_mode="fake" if fake else "live")
    if engine.llm is None and not offline:
        typer.echo("LLM_BASE_URL not set: running offline (numeric pipeline + template)")
    request = PredictRequest(serial_number=serial,
                             as_of_date=date.fromisoformat(as_of) if as_of else None,
                             horizons=cfg.horizons, cost_ratio=cost_ratio, offline=offline)
    resp = engine.predict(request)
    if as_json:
        typer.echo(resp.model_dump_json(indent=2))
        return
    ctx = resp.context_summary
    typer.echo(f"run {resp.run_id}: {serial} ({ctx.get('model')}) as of {ctx.get('as_of_date')}"
               f" | LLM available: {resp.llm_available} ({resp.llm_calls} calls) | "
               f"{resp.latency_ms:.0f} ms")
    for err in resp.errors:
        typer.echo(f"  error: {err}")
    if resp.fused:
        probs = ", ".join(f"{h}d {p:.2%}" for h, p in sorted(resp.fused.probabilities.items()))
        typer.echo(f"fused risk: {probs} | uncertainty {resp.fused.uncertainty:.2f} | "
                   f"disagreement {resp.fused.disagreement:.2f}")
    for f in resp.findings:
        risk = ", ".join(f"{h}d {p:.2%}" for h, p in sorted(f.risk.items()))
        typer.echo(f"  [{f.status:8}] {f.agent:20} {risk:32} {f.summary[:110]}")
    if resp.decision:
        costs = ", ".join(f"{a} {c:.2f}" for a, c in resp.decision.expected_costs.items())
        typer.echo(f"decision: {resp.decision.action} (expected costs: {costs})")
        for r in resp.decision.reasons:
            typer.echo(f"  - {r}")
    if resp.explanation:
        typer.echo(f"explanation ({resp.explanation.source}): {resp.explanation.text}")


if __name__ == "__main__":
    app()
