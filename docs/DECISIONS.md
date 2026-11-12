# Decisions and deviations

Every place where the build differs from `PROJECT_BUILD_PROMPT.md`, or where the spec left a
choice open, with the reason. Newest entries are appended at the end of each section.

## Environment and dependencies

- **PyPI mirror during development.** `pypi.org` / `files.pythonhosted.org` were unreachable from
  the development network (connections timed out even outside the sandbox), while npm, Hugging
  Face, PyTorch and Backblaze were reachable. Packages were installed from the Tsinghua PyPI
  mirror (`https://pypi.tuna.tsinghua.edu.cn/simple`, session-only, via `PIP_INDEX_URL`) and
  `torch` from the official PyTorch CPU index. The pinned versions in `requirements.txt` are
  ordinary PyPI versions; a normal `pip install -r requirements.txt` works on any network that
  can reach PyPI.
- **Newer major versions than the spec author may assume.** Pinned: pandas 3.1, DuckDB 2.0,
  LangChain core 1.7, LangGraph 1.2, openai 3.37, transformers 5.20. APIs were checked against
  the installed versions (e.g. `ChatOpenAI` exposes `base_url` as an alias of
  `openai_api_base`, `max_tokens` as an alias of `max_completion_tokens`; transformers 5 uses
  `dtype=` instead of `torch_dtype=`).
- **Non-ASCII project path.** The repository lives under `ドキュメント`. Windows consoles default to
  cp1252, so CLI entry points reconfigure stdout/stderr to UTF-8 (`scripts/_common.py`). Native
  libraries that take file paths (FAISS, LightGBM) are given bytes/strings through Python file
  handles instead of paths (see the models and RAG sections).

## Colab LLM server (`qwen35_colab_server.ipynb`)

- **Model facts verified** on the Hugging Face model card (Oct 2026): `Qwen/Qwen3.5-4B` exists,
  architecture `Qwen3_5ForConditionalGeneration` (vision-language), chat template honours
  `enable_thinking`; vLLM serves it with `--language-model-only` for text-only use; thinking is
  disabled with `chat_template_kwargs={"enable_thinking": false}`.
- **Proxy in front of every backend.** Instead of relying on each backend's auth and reasoning
  parser, a small FastAPI proxy (public port) enforces the API key, injects the thinking default,
  strips `<think>` blocks and `reasoning_content` (streaming and non-streaming, including the case
  where the template pre-fills the opening `<think>` so only `</think>` appears), and keeps request
  stats for the monitor cell. vLLM therefore runs **without** `--reasoning-parser`, so stripping
  is deterministic. The backend listens on `PORT+1` on localhost with an internal key.
- **Fallback loads the language model only** via `AutoModelForCausalLM` (transformers maps
  `qwen3_5` to `Qwen3_5ForCausalLM`); its gated-delta-rule layers have pure-PyTorch
  implementations, so no extra kernels are required on a T4.
- **Verification limits.** No GPU is available on the development machine, so the notebook was
  validated structurally (nbformat validation, every cell parses) and the proxy and fallback
  server were tested locally: the proxy against a fake backend (auth, stripping, streaming,
  stats) and the fallback on CPU with `HuggingFaceTB/SmolLM2-135M-Instruct` behind the proxy
  using the `openai` client. vLLM 0.32 on a T4 (compute capability 7.5) could not be tested here;
  if it fails, the notebook switches to the fallback automatically.
- Cell headers follow the spec numbering 1–12. The "Serve" step has three extra `%%writefile`
  cells (proxy, fallback server, validator), and a manual Troubleshooting cell follows Shutdown.
- **Revision after a first Colab run returned an opaque HTTP 500** in the thinking-control
  cell:
  - vLLM now runs in its own `uv` virtual environment (Python 3.12), so it cannot replace
    Colab's torch/transformers that the fallback needs. It starts with `--enforce-eager`
    and `--max-num-seqs 16`, which are safer on a T4.
  - A backend is accepted only after a strict validation: the arithmetic answer must contain
    42, plus JSON, thinking mode, streaming and a 400-token generation. Passing a single short
    request is no longer enough.
  - If vLLM is rejected, the GPU is freed and the fallback is validated the same way. If
    neither passes, the notebook stops with the validation table and log tails.
  - The proxy and the fallback return JSON errors that name the cause (backend status and
    message, or the exception), never a bare 500.
  - The fallback generates off the event loop, guards against fp16 inf/nan logits
    (`remove_invalid_values`), and loads the full vision-language class if the text-only
    class fails. It refuses a model with missing weights.
  - Verified locally on CPU with `tiny-random/qwen3.5`: both loading paths load with 0
    missing weights, the full validation passes directly and through the proxy, and the
    error paths return 400/401/502 with readable messages.

## Data

- **Hugging Face mirror is stale.** `backblaze/Drive_Stats` on Hugging Face only contains zips up
  to 2023 Q2, so the official Backblaze page is the default source. The mirror stays available
  (`data.source: huggingface`) for older years.
- **Default years 2024–2025.** The two most recent complete years published on the Drive Stats
  page at build time (2026 Q1–Q2 also exist, but 2026 is incomplete).
- **Slow, bursty download link.** The Backblaze B2 host stalled often from the development
  network, so the downloader resumes with HTTP range requests, retries for as long as bytes keep
  arriving, and fetches several quarters in parallel (`data.download_workers`).
- **Two-stage harmonization.** Choosing the top models needs failure and drive-day counts over
  the whole period, so every daily CSV is first written to a per-day staging Parquet file
  (all models, configured columns only; resumable). Model selection and the final
  `model/year/quarter` Parquet are then done in DuckDB. Duplicate `(serial, date)` rows are
  removed, keeping the row with `failure = 1`. The final write runs one quarter at a time: a
  single global dedup + sort over all ~100M rows spilled 25 GB to disk and was I/O-bound
  (an estimated 3 h); per quarter it takes about 1–1.5 min. Duplicates are per day, so they
  never cross quarters.
- **Model selection rule.** Models with at least `min_failures_per_model` failures, ranked
  by failures, then drive-days. The first `top_k_models` are used for training; the next
  `holdout_models` are held out entirely as the unseen-model test set.
- **Censoring.** A drive that disappears without a failure row has an unknown outcome
  after its last snapshot, so labels whose horizon passes that date are null and are excluded
  per horizon. Windows on or after the failure day are not sampled.
- **Splits combine serial hashing and time periods.** Serials are md5-hashed into
  train/val/test (60/20/20) *and* each split samples only from its own time period (60/20/20
  of the date range). A window's whole label horizon must also lie inside its period, so
  training labels never look into the validation or test period. The `unseen` split uses the
  test period.
- **Natural-distribution evaluation is a uniform sample.** Evaluating every drive-day is
  infeasible on a laptop, so validation/test/unseen are uniform samples capped at
  `eval_max_windows`. Uniform sampling keeps the natural class prior. Lead time uses
  separate, uncapped sets: every positive window of the failing test and unseen drives.
- **Feature engineering in DuckDB.** Window features are computed in SQL (range joins plus
  `regr_slope`, `arg_max`, filtered aggregates). The same function serves training tables and
  single-drive inference. "Days since first non-zero" uses all history up to `as_of`. A test
  recomputes features on a view truncated at `as_of` and requires identical results.

## Models

- **Early stopping uses a slice of the training split** (serial-grouped 15%), not
  validation. Validation is reserved for calibration and fusion, and those use 5-fold
  serial cross-fitting so fusion is trained on out-of-fold calibrated risks.
- **Sequence model.** A 1D-CNN by default (an LSTM is configurable), trained on a capped set of
  windows (`models.sequence.max_windows`, all positives kept) to fit in RAM. It predicts all
  horizons with a masked loss and is calibrated on a uniform validation sample.
- **Monotone horizons.** Per-horizon models and calibrators can produce
  P(≤14d) < P(≤7d), which is logically impossible. All calibrated outputs are corrected
  with a running maximum over horizons.
- **Fusion with missing agents.** Disabled or failed agents are imputed with the model's
  base rate ("no evidence"), so the trained fusion stays valid. Ablations that remove an
  agent instead *refit* fusion without it on the stored out-of-fold validation risks.
- **Uncertainty and disagreement.** Disagreement is 2 × the weighted standard deviation of the
  agents' calibrated risks at the longest horizon, using the fusion's non-negative,
  normalized agent weights. The first version used max − min. On real data it escalated
  almost every risky drive, because specialists *should* differ (a media failure need not
  show in thermal data) and low-weight agents always sit near the base rate.
  Uncertainty is 1 − (1 − H(p))·coverage, where H is the
  normalized binary entropy of the fused probability and coverage is the observed fraction
  of the window times the fraction of numeric agents that contributed. These definitions are
  ours; the spec only names the thresholds.
- **Unseen drive models** use pooled base rates, pooled normalization statistics and the
  pooled drift reference.
- **Drift features exclude levels of cumulative counters.** Levels of lifetime counters
  (power-on hours, LBAs written, error totals) rise as every drive ages, so PSI on them
  flagged normal aging as drift (PSI 1.45 for usage/wear on the first real run). Drift is
  monitored on rate statistics (delta, slope, acceleration, increases) for cumulative raw
  attributes, and on levels only for normalized values and non-cumulative raw attributes
  (temperature 190/194, pending sectors 197).
- **Thread safety.** LightGBM boosters are not safe for concurrent `predict`. Graph agents
  run in parallel threads and the API serves concurrent requests, and the first full
  evaluation crashed natively during parallel graph runs. Each model now serializes its own
  predictions with a lock (different models still run in parallel).
- **Artifacts without pickle.** LightGBM is saved with `model_to_string`, FAISS with
  `serialize_index`, torch through file handles, and calibrators and fusion as JSON. This is
  safe on the non-ASCII path and avoids loading untrusted pickles.

## RAG

- **Case memory produces a calibrated risk, not only explanations.** The case agent
  reports the similarity-weighted share of its k nearest *training* drives (one per serial,
  same drive model) that failed within each horizon. This share is shrunk toward the base
  rate, calibrated on validation (cross-fitted), and used by a retrieval-aware fusion fitted
  on out-of-fold validation risks. Without this, retrieval could not affect prediction and
  H2's prediction claim would be untestable. The `no_case_retrieval` ablation removes it.
- **Knowledge base.** Seven authored documents, chunked one snippet per `##` section,
  embedded with `BAAI/bge-small-en-v1.5` on CPU (with the model's query instruction). Every
  factual claim cites Wikipedia's SMART article, the Backblaze SMART-stats blog post (2016),
  the Google FAST'07 study (as summarized by Wikipedia) or the Backblaze dataset card. These
  sources were fetched and checked during the build.

## Agents and graph

- **One generic specialist prompt** (`specialist.md`) plus `planner`, `explainer` and
  `repair` prompts. An agent-specific prompt is used automatically if
  `prompts/agent_<name>.md` exists. Every prompt starts with `ROLE:` for routing in FakeLLM
  and logs.
- **Retrieval agents and the retrieve node.** `case_retrieval` and `knowledge_retrieval` run
  in the parallel dispatch like any agent (their findings cite case and snippet ids). The
  `retrieve` node then materializes `retrieved_cases` and `knowledge` for the explainer,
  using a knowledge query focused on the riskiest agents' findings.
- **Single-agent baseline.** Implemented as a `global`-kind agent over all attributes,
  using the all-features model's risk without fusion. Its *numeric* metrics therefore equal
  `lgbm_global` by construction; it differs only in graph runs.
- **The planner may drop agents**, but a first-attempt plan with no numeric agent is
  rejected (default plan). On a replan, the default is to rerun only the failed agents.
- **Explainer facts state the decision logic explicitly.** In the first live run, Qwen
  wrote that escalation "is the lowest expected cost action". It isn't: escalation always
  costs review plus the best action. The facts block now separates "lowest expected-cost
  action" from "escalated because of these rules", and the prompt requires both and asks for
  cost numbers to be copied exactly. The next live run explained it correctly.
- **Explanation validation.** Cited ids are kept only if they exist in the run, invented
  bracketed ids are removed from the text, and an LLM explanation with no valid citation
  falls back to the template.
- **Run-time dependencies are LangGraph `context`** (not state), so checkpoints contain only
  Pydantic contracts. Those contracts are allow-listed for msgpack deserialization
  (`allowed_msgpack_modules`).
- **Escalation is never chosen on cost alone.** Its expected cost is review cost plus the
  best other action, so only the rules (uncertainty, disagreement, drift alert) trigger it.
  The suggested action is kept in the reasons for the reviewer.

## API and client

- **Extra endpoints**: `/drives/stats` (KPIs and risk histogram for the overview),
  `/runs/{id}`, `/escalations` (the queue for the Escalations page) and `/settings/test`
  (connection test with unsaved values).
- **Fleet snapshot.** The drive list needs a risk for every drive; computing it per request is
  too slow. `scripts.index` scores each drive's latest window with the numeric pipeline
  (no LLM) and stores `outputs/artifacts/fleet_snapshot.parquet`. Full multi-agent runs
  override the action shown for a drive.
- **Settings persistence.** UI settings (including the API key) are stored in
  `outputs/runtime/settings.json` (git-ignored). Reports always mask the key.
- **Client versions.** TypeScript 5.9 and React Router 7 instead of the newest majors
  (TypeScript 7 native compiler, React Router 8), whose APIs could not be verified here;
  Tailwind CSS 4 through `@tailwindcss/vite`.

## Evaluation protocol

- **Systems and ablations** are listed in `evaluation/baselines.py` and
  `evaluation/ablations.py`. `no_knowledge_retrieval` and `single_llm_agent` match
  `full_system` and `lgbm_global` numerically by design (stated in every report).
- **Drift-aware recalibration (H4).** For each drive model and test month, if PSI drift vs the
  validation period is at least `warn`, a Platt map is refit on that model's most recent
  windows whose labels had matured before the month began (validation windows plus earlier
  months). Label delay is respected; no label from the evaluated month is used. Lead-time
  sets reuse the same maps.
- **Costs.** Mean realized cost per window under two stated assumptions for escalated cases:
  the reviewer applies the policy's suggested action (`model`, conservative), or the
  hindsight-best action (`oracle`, optimistic). Cost without escalation, always-monitor and
  the hindsight optimum are reported as references.
- **Statistics.** Mean ± std over seeds. 95% CIs and paired p-values against the best
  baseline (by AUPRC) come from a cluster bootstrap by drive serial on the first seed's
  predictions, with shared resampling weights across systems. Weighted AUPRC is computed in
  O(n) per resample and matches scikit-learn's value with unit weights (tested).
- **Numeric results are written first.** The runner writes the full numeric report before it
  starts the slow graph runs, then rewrites it with the graph-run table, so a failure in the
  graph phase cannot lose the numeric results. Graph runs share one set of loaded resources
  (data, models, indexes, drift cache) across systems. Offline runs are sequential because
  they are bound by local DuckDB work.
- **LLM sample.** Graph runs use a stratified sample (all positives up to a cap, then random
  negatives) with the first seed's models. If the configured live endpoint is unreachable,
  the runner switches to offline mode and says so in the report: those are not LLM results.
  Explanation metrics: share of LLM-written explanations, share with at least one valid
  citation, mean citations, and the share of numbers in the text that match a number in the
  evidence given to the explainer (a faithfulness proxy).
