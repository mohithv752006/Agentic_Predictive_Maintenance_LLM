"""Route registry: add an endpoint group = one module here + one entry in ROUTERS."""

from pdm_agents.api.routes import drives, experiments, feedback, health, predict, settings

ROUTERS = [
    health.router,
    settings.router,
    drives.router,
    predict.router,
    experiments.router,
    feedback.router,
]
