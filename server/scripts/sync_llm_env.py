"""Copy the LLM connection saved on the dashboard's Settings page into server/.env.

Command-line jobs (evaluation, predict) read LLM_BASE_URL / LLM_API_KEY from .env, while the
dashboard saves them to outputs/runtime/settings.json. Run this after saving new Colab
settings on the dashboard. The API key is never printed.

Usage:
  python -m scripts.sync_llm_env
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from pdm_agents.llm.health import check_health
from scripts._common import bootstrap

app = typer.Typer(add_completion=False)
KEYS = {"llm_base_url": "LLM_BASE_URL", "llm_api_key": "LLM_API_KEY", "llm_model": "LLM_MODEL"}


@app.command()
def main(env_file: Path = typer.Option(Path(".env"), help="dotenv file to update")) -> None:
    cfg = bootstrap()
    path = cfg.paths.runtime / "settings.json"
    if not path.exists():
        raise typer.BadParameter(f"{path} not found; save the LLM settings on the dashboard first")
    saved = json.loads(path.read_text(encoding="utf-8"))
    values = {env: str(saved[k]) for k, env in KEYS.items() if saved.get(k)}
    if "LLM_BASE_URL" not in values or "LLM_API_KEY" not in values:
        raise typer.BadParameter("the dashboard settings have no LLM URL or API key")
    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    kept = [ln for ln in lines if ln.split("=", 1)[0].strip() not in values]
    env_file.write_text("\n".join([*kept, *(f"{k}={v}" for k, v in values.items())]) + "\n",
                        encoding="utf-8")
    typer.echo(f"wrote {', '.join(values)} to {env_file.resolve()}")
    health = check_health(cfg.llm.model_copy(update={
        "base_url": values["LLM_BASE_URL"], "api_key": values["LLM_API_KEY"],
        "model": values.get("LLM_MODEL", cfg.llm.model)}))
    if health.reachable:
        typer.echo(f"LLM reachable at {values['LLM_BASE_URL']} ({health.latency_ms:.0f} ms)")
    else:
        typer.echo(f"LLM NOT reachable at {values['LLM_BASE_URL']}: {health.error}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
