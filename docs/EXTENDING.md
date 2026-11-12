# Extending the system

Each kind of extension touches **one new file plus one registry or config entry**.

| Add / remove | New file | Entry |
|---|---|---|
| Agent | `server/src/pdm_agents/agents/<name>.py` (subclass `SpecialistAgent` or `FamilyAgent`) | `server/configs/agents.yaml` |
| Tool | function or factory in `server/src/pdm_agents/tools/<module>.py` | `tools/registry.py` → `build_tools` |
| Model | `server/src/pdm_agents/models/<name>.py` (fit / predict_proba / save / load) | `models/registry.py` → `MODEL_REGISTRY` |
| Evaluation system / ablation | scoring function | `evaluation/baselines.py` → `BASELINES` or `evaluation/ablations.py` |
| Experiment | `server/configs/experiments/<name>.yaml` | none |
| API route group | `server/src/pdm_agents/api/routes/<name>.py` | `api/routes/__init__.py` → `ROUTERS` |
| Frontend page | `client/src/pages/<Name>.tsx` | `client/src/routes.tsx` → `PAGES` |
| Prompt for one agent | `server/src/pdm_agents/llm/prompts/agent_<name>.md` | none (picked up automatically; otherwise `specialist.md` is used) |

Removing something is the reverse. For agents, set `enabled: false` in `agents.yaml`; the graph builder, planner, fusion and UI all follow.

## Adding an agent: worked example (proven by a test)

**1. One file.** This is `server/tests/dummy_agent.py`; a real agent would live in `src/pdm_agents/agents/`.

```python
from pdm_agents.agents.base import SpecialistAgent
from pdm_agents.schemas import DriveContext, EvidenceItem, ToolEvidence


class DummyAgent(SpecialistAgent):
    def run_tools(self, ctx: DriveContext) -> ToolEvidence:
        n = len(ctx.window.dates)
        return ToolEvidence(agent=self.name, metrics={"days": float(n)},
                            evidence=[EvidenceItem(id=f"E-{self.name}-0", kind="note",
                                                   label="days observed", value=n)])
```

**2. One YAML entry** in `configs/agents.yaml`:

```yaml
  dummy:
    class: tests.dummy_agent:DummyAgent
    enabled: true
    kind: context            # numeric | context | retrieval | global
    description: Counts observed days.
```

No other edits are needed:
- `agents/registry.py` imports the class from the `class` path;
- `graph/builder.py` adds an `agent_dummy` node for every enabled agent;
- the planner offers it to the LLM;
- `/settings` lists it as a toggle;
- the UI timeline shows it.

`server/tests/test_graph.py::test_dummy_agent_added_with_one_file_and_one_yaml_entry` proves it. The test adds exactly this YAML entry as a config override, runs a full prediction, and asserts that the `dummy` finding has status `ok` and that an `agent_dummy` node appears in the run trace. Run it with:

```bash
cd server && pytest tests/test_graph.py -k dummy -q
```

### Agent kinds

- `numeric`: owns a SMART attribute family. `scripts.train` trains a calibrated LightGBM per horizon for it, and its risk enters fusion. Its attributes must be harmonized columns (`smart_<id>_raw|normalized`) listed in `data.smart_attributes`. If you add new attributes, re-run `scripts.prepare`; features are computed for every numeric agent in `agents.yaml`, enabled or not. Subclass `FamilyAgent` and override `domain_notes` for agent-specific observations.
- `context`: adds evidence; its risk, if any, is shown but not fused.
- `retrieval`: a RAG agent, skipped when `features.use_rag` is false.
- `global`: one agent over all attributes using the all-features model. Its risk is used directly, without fusion; the single-agent baseline uses this kind.

Every agent inherits the contract: `run()` never raises, a tool failure gives `status="failed"`, and LLM failure gives a template summary with `status="degraded"`.

## Adding a tool

```python
# tools/my_tools.py
from langchain_core.tools import tool
from pydantic import BaseModel

class Args(BaseModel):
    values: list[float | None]

@tool("volatility", args_schema=Args)
def volatility(values: list[float | None]) -> float:
    """Standard deviation of the non-missing values."""
    ...
```

Then add `tools["volatility"] = volatility` in `tools/registry.py`. Agents call it with `self.res.tool("volatility").invoke({...})`. If a tool's resources are missing, it is left out of the registry, and agents that need it fail cleanly.

## Adding a model

Implement `fit`, `predict_proba`, `save` and `load` (see `models/base.py::ProbabilityModel`), register it in `models/registry.py`, and reference it from training code or an evaluation system. Save artifacts through Python file handles, not native path APIs: the repository path may contain non-ASCII characters.

## Adding an experiment, baseline or ablation

An experiment is a YAML file:

```yaml
experiment:
  name: my_experiment
  systems: [full_system, my_ablation]
  sets: [test, unseen]
  llm_mode: live        # live | offline | fake
  run_llm_sample: false
overrides:              # merged over default.yaml for this run
  decision: {cost_ratios: [20, 100]}
```

A new system is a scoring function that returns a `SystemOutput` (probabilities per horizon, uncertainty, disagreement), plus one entry in `BASELINES` or `ablations()`. If the system has a `graph` override dict, the runner also runs it through the real graph on the stratified LLM sample.

## Adding a frontend page

Create `client/src/pages/MyPage.tsx` with a default-exported component, then add `{ path: "/my-page", label: "My page", component: MyPage, nav: true }` to `PAGES` in `client/src/routes.tsx`. Navigation and routing are built from that list.
