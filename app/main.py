from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.baseline import run_baseline
from app.env import Environment
from app.grader import grade_state
from app.models import Action, GraderRequest, GraderResponse, Observation, ResetRequest, State, StepResult
from app.tasks import TaskDefinition, list_tasks


app = FastAPI(
    title="ai_workops_env",
    description="Realistic AI workplace simulation environment (OpenEnv-style API).",
    version="0.1.0",
)

# In-memory singleton environment (no external DB).
ENV = Environment(env_name="ai_workops_env", max_steps=8)


@app.exception_handler(ValueError)
def _value_error_handler(_request: Any, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "invalid_request", "detail": str(exc)})


@app.get("/")
def health() -> Dict[str, str]:
    """
    Basic health endpoint for container platforms (including HF Spaces).
    """

    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest) -> Observation:
    """
    Reset the environment.
    - If req.task_id is provided, resets into that task only.
    - If req.difficulty is provided, picks a task by difficulty.
    - Otherwise loads the canonical task set.
    """

    try:
        return ENV.reset(task_id=req.task_id, difficulty=req.difficulty)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    """
    Take one environment step.
    """

    # Pydantic validates structure; env validates semantic constraints.
    return ENV.step(action)


@app.get("/state", response_model=State)
def state() -> State:
    """
    Return full internal environment state.
    """

    return ENV.state()


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    """
    List available tasks and core schemas (OpenEnv-style discovery).
    """

    ts: List[TaskDefinition] = list_tasks()
    return {
        "tasks": [t.model_dump() for t in ts],
        "schemas": {
            "action": Action.model_json_schema(),
            "observation": Observation.model_json_schema(),
            "step_result": StepResult.model_json_schema(),
            "state": State.model_json_schema(),
        },
    }


@app.post("/grader", response_model=GraderResponse)
def grader(req: GraderRequest) -> GraderResponse:
    """
    Grade a completed episode (or partial transcript) into a 0..1 score.
    If req.history is omitted, grades the current ENV state history.
    """

    st = ENV.state()
    score, details = grade_state(st, task_id=req.task_id, history=req.history)
    return GraderResponse(score=score, details=details)


@app.get("/baseline")
def baseline() -> Dict[str, Any]:
    """
    Run the baseline agent across all tasks and return reproducible scores.
    """

    raw = run_baseline(max_steps=ENV.max_steps)
    task_scores = {r["task_id"]: float(r["score"]) for r in raw["results"]}
    return {"task_scores": task_scores, "average_score": float(raw["average_score"])}

