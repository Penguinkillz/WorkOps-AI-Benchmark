from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    """
    OpenEnv action payload.

    Keep this intentionally generic: different tasks can use different "type"
    values and optional fields.
    """

    type: str = Field(..., description="Action type identifier (e.g. reply, ignore, escalate, resolve, refund).")
    task_id: Optional[str] = Field(default=None, description="Target task id (defaults to current episode task).")
    content: Optional[str] = Field(default=None, description="Optional free-form content (e.g. reply text).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional action metadata.")


class InboxItem(BaseModel):
    id: str
    kind: Literal["email", "ticket", "workflow"]
    subject: str
    body: str
    difficulty: Literal["easy", "medium", "hard"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """
    OpenEnv observation payload.
    """

    step: int
    current_task_id: str
    inbox: List[InboxItem]
    last_action: Optional[Action] = None
    message: str = Field(..., description="Human-readable observation summary.")


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Reward increment for the last step (0..1).")
    components: Dict[str, float] = Field(default_factory=dict, description="Reward breakdown for transparency.")


class EnvMetadata(BaseModel):
    timestep: int = 0
    max_steps: int = 8
    priority: Literal["low", "normal", "high"] = "normal"


class State(BaseModel):
    """
    Full in-memory environment state. Returned by GET /state.
    """

    env_name: str
    episode_id: str
    current_task_id: str
    inbox: List[InboxItem]
    history: List[Action]
    metadata: EnvMetadata
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default=None, description="Reset into a specific task id.")
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(default=None, description="Pick a task by difficulty.")


class GraderRequest(BaseModel):
    """
    Minimal episode evaluation request.

    The environment is in-memory, so callers can either:
    - omit fields and grade the current episode state, or
    - pass an explicit transcript for offline evaluation.
    """

    task_id: Optional[str] = None
    history: Optional[List[Action]] = None


class GraderResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
