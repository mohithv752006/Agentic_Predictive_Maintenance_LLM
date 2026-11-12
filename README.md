# Drive Failure Agents

Cost-aware multi-agent RAG for hard drive failure prediction on the Backblaze Drive Stats data.

For each drive and date, the system takes the drive's last 30 days of SMART snapshots and returns:

- calibrated probabilities that the drive fails within 7, 14 and 30 days;
- a recommended action: `monitor`, `schedule_replacement`, `replace_now` or `escalate_to_human`. It is the action with the lowest expected cost for a configurable cost ratio (missed failure ÷ planned replacement);
- an explanation that cites the specialist agents' findings, similar past drives and knowledge snippets by id.

Probabilities always come from trained, calibrated models. The LLM (Qwen3.5-4B, served from Google Colab) only plans, interprets and explains. If the LLM is unreachable, the system still returns probabilities and a decision, adds a template explanation, and reports `llm_available: false`.

```
Browser (client) ──REST/SSE──> FastAPI (server) ──LangGraph──> agents / tools / RAG
                                                     └──HTTPS (OpenAI-compatible)──> Colab: Qwen3.5-4B
```

| Part | What it is |
|---|---|
| `qwen35_colab_server.ipynb` | Serves Qwen3.5-4B on a Colab GPU behind a key-protected, OpenAI-compatible tunnel |
| `server/` | Python package `pdm_agents`: data pipeline, models, LangGraph agents, RAG, decisions, evaluation, FastAPI |
| `client/` | React + TypeScript dashboard (fleet, drive analysis with live agent timeline, experiments, escalations, settings) |
| `docs/` | [ARCHITECTURE](docs/ARCHITECTURE.md), [EXTENDING](docs/EXTENDING.md), [DECISIONS](docs/DECISIONS.md) |

## Requirements

- Python 3.11+ (developed with 3.12), Node.js 20.19+ or 22.12+.
- About 25 GB of free disk: 8.5 GB of raw zips, about 6 GB of staging Parquet, the final Parquet, and DuckDB spill space during preparation.
- 16 GB of RAM is enough; DuckDB is capped at `data.duckdb_memory_limit` and spills to disk.
- No GPU is needed locally. The LLM runs on Colab.

## 1. Start the LLM on Colab

1. Open `qwen35_colab_server.ipynb` in Google Colab.
2. Choose **Runtime → Change runtime type → T4 GPU** (or better).
3. Optionally add Colab **Secrets** (the key icon): `HF_TOKEN`; `NGROK_TOKEN`, only if you pick the ngrok tunnel; and `LLM_API_KEY`, to keep the same key across sessions.
4. Choose **Runtime → Run all**. The notebook installs vLLM. If vLLM fails to start or fails its self-test, the notebook switches to a plain `transformers` server automatically. Self-tests print PASS/FAIL.
5. Copy the printed block into `server/.env`, or paste it on the web app's **Settings** page:
   ```
   LLM_BASE_URL=https://<tunnel-host>/v1
   LLM_API_KEY=<key>
   LLM_MODEL=Qwen/Qwen3.5-4B
   ```

The tunnel URL changes every session, so update `LLM_BASE_URL` each time. Free Colab sessions also disconnect when idle; leave the monitor cell running and the tab open.

## 2. Server (VS Code, local)

```bash
cd server
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env                                   # paste the LLM_* values from Colab
python -m scripts.download && python -m scripts.prepare && python -m scripts.train && python -m scripts.index
python main.py                                         # FastAPI on http://127.0.0.1:8000 (same as python -m scripts.serve)
python -m scripts.evaluate --experiment configs/experiments/full.yaml
```

What each step does:

| Command | Output |
|---|---|
| `scripts.download` | Finds the configured quarters on the official Drive Stats page and downloads them in parallel, with resume (`data/raw/`). Add `--list-only` to preview. |
| `scripts.prepare` | Streams each daily CSV out of the zips, harmonizes changing SMART schemas, selects drive models (top 4 by failures + 1 held-out), writes partitioned Parquet, then builds leakage-safe labels, splits and window features (`data/parquet/`). Prints a data summary. |
| `scripts.train` | Per seed: LightGBM per SMART family and on all features, the sequence model, calibrators, fusion, base rates and the drift reference (`outputs/artifacts/models/seed_<n>/`). |
| `scripts.index` | Case memory (training drives only), calibrated case risk with retrieval-aware fusion, the knowledge index, and the fleet snapshot behind the dashboard. |
| `scripts.serve` | The API. Open `http://127.0.0.1:8000/docs` for the interactive schema. |
| `scripts.predict --serial <S>` | One full multi-agent run in the terminal. Add `--offline` (no LLM), `--fake` (deterministic FakeLLM) or `--json`. |
| `scripts.evaluate --experiment <yaml>` | Baselines, ablations, cost sweep and drift experiments. Reports go to `outputs/reports/<id>/`. |

Tests: `pytest` (it builds a small synthetic dataset; no download or LLM needed). Lint: `ruff check .`.

> Windows note: `torch` is pinned to the CPU build. `requirements.txt` already adds the PyTorch CPU index (`--extra-index-url https://download.pytorch.org/whl/cpu`).

## 3. Client

```bash
cd client && npm install && cp .env.example .env && npm run dev
```

Then open `http://127.0.0.1:5173`. `VITE_API_BASE_URL` points the client at the API (default `http://127.0.0.1:8000`).

Pages:
- **Fleet overview:** KPI cards, risk distribution, sortable and filterable drive table.
- **Drive analysis:** SMART charts per family, and a *Run analysis* button with a cost-ratio slider. While the run streams (SSE), it shows a live agent timeline, fused probabilities, the decision with the expected cost of every action, and an explanation whose citations jump to the cited finding, similar drive or knowledge snippet.
- **Experiments:** report tables and plots.
- **Escalations:** approve or override escalated cases; the feedback is stored.
- **Settings:** LLM endpoint, key and model, a connection test, agent toggles, default cost ratio.

A banner appears whenever the LLM is offline (degraded mode).

## Running the server pipeline inside Colab instead

If your laptop cannot handle data preparation, run the Python pipeline in Colab; a CPU runtime is enough for everything except the LLM. Environment detection is automatic: inside Colab, all data and output paths move to `/content/drive/MyDrive/drive-failure-agents/`, and secrets are read from Colab Secrets.

```python
from google.colab import drive; drive.mount("/content/drive")
!git clone <your fork of this repository> /content/drive-failure-agents
%cd /content/drive-failure-agents/server
!pip install -q -r requirements.txt && pip install -q -e .
!python -m scripts.download && python -m scripts.prepare && python -m scripts.train && python -m scripts.index
```

Afterwards, either serve from Colab or copy `data/parquet/` and `outputs/` from Google Drive into `server/` on your laptop and run `python -m scripts.serve` locally.

## Configuration

All settings live in `server/configs/`:
- `default.yaml`: paths, data scope, horizons, LLM, RAG, models, decision, drift, evaluation, feature flags, seeds.
- `agents.yaml`: which agents are enabled and their SMART attributes.
- `actions.yaml`: the action set and cost table.
- `experiments/*.yaml`: one file per experiment or ablation.

`${VAR}` values come from `.env` or Colab Secrets. Settings changed on the UI's Settings page are stored in `outputs/runtime/settings.json`. That file holds the API key in plain text; it is git-ignored, so keep it local.

## Data source and attribution

This project uses the **Backblaze Drive Stats** dataset: <https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data>.
**Data source: Backblaze.** As Backblaze asks, we cite Backblaze as the source, we are solely responsible for how we use the data, and the data itself is not sold.

The default scope is 2024–2025, the two most recent complete years at build time. The drive models are selected from the data by the pipeline, not hard-coded.

## Results

Evaluation numbers are produced only by `scripts.evaluate` and written to `outputs/reports/`; this README does not repeat or invent them. See [DECISIONS](docs/DECISIONS.md) for the evaluation protocol and known limitations.
