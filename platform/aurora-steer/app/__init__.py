"""`aurora-steer` Google ADK 2.0 agent package and FastAPI steering service.

`google-adk` is imported lazily (only by `app.agent` / `app.runner` when an LLM
call is made), so the API, schemas, guardrails and OKF loader can be imported
and tested without it. `adk web` / `adk run` still discover `app.agent.root_agent`
through ADK's submodule loader.
"""

__all__: list[str] = []
