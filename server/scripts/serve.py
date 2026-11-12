"""Start the FastAPI server.

Usage: python -m scripts.serve [--host 127.0.0.1] [--port 8000] [--config extra.yaml]
"""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from pdm_agents.api.main import create_app
from scripts._common import bootstrap

app = typer.Typer(add_completion=False)


@app.command()
def main(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8000, help="Port"),
    config: Path | None = typer.Option(None, help="Extra YAML merged over default.yaml"),
) -> None:
    cfg = bootstrap(config_file=config)
    uvicorn.run(create_app(cfg), host=host, port=port, log_level=cfg.logging.level.lower())


if __name__ == "__main__":
    app()
