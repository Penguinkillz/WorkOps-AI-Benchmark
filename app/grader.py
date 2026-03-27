from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.models import Action, State
from app.tasks import TaskDefinition, get_task


def _action_matches_expected(action: Action, expected_type: str, content_contains: Optional[str]) -> bool:
    if action.type != expected_type:
        return False
    if content_contains is None:
        return True
    return content_contains.lower() in (action.content or "").lower()

def _importance_multiplier(task: TaskDefinition) -> float:
    hidden = dict(task.metadata.get("hidden", {})) if isinstance(task.metadata, dict) else {}
    vip = bool(hidden.get("vip_flag", False))
    urgency = str(hidden.get("urgency", "normal")).lower()
    risk = str(hidden.get("risk_level", "medium")).lower()

    factor = 1.0
    if vip:
        factor *= 1.25
    if urgency in ("high", "urgent"):
        factor *= 1.20
    elif urgency in ("medium",):
        factor *= 1.10
    if risk in ("high",):
        factor *= 1.15
    return factor


def _repeat_penalty(history: List[Action]) -> float:
    if len(history) < 2:
        return 0.0
    repeats = 0
    for i in range(1, len(history)):
        if history[i].type == history[i - 1].type and history[i].task_id == history[i - 1].task_id:
            repeats += 1
    # Cap to keep score in [0,1]
    return min(0.15, 0.03 * float(repeats))


def _unnecessary_action_penalty(task: TaskDefinition, history: List[Action]) -> float:
    """
    Penalize actions that do not advance the expected sequence (manager sees as noise).
    """

    expected = task.expected
    if not expected:
        return 0.0
    idx = 0
    noise = 0
    for a in history:
        if a.task_id not in (None, task.id):
            continue
        if idx < len(expected) and _action_matches_expected(a, expected[idx].type, expected[idx].content_contains):
            idx += 1
        else:
            noise += 1
    return min(0.20, 0.04 * float(noise))


def grade_task(task: TaskDefinition, history: List[Action]) -> Tuple[float, Dict[str, Any]]:
    """
    Deterministically grade a single task transcript.

    Scoring (0..1):
    - correctness: actions match expected in order (0..1)
    - completion: did agent finish the sequence (0/1)
    - efficiency: steps used vs optimal (0..1)
    - priority handling: importance-weighted behavior (0..1)
    - explicit penalties for repeats and unnecessary actions
    """

    expected = task.expected
    if not expected:
        return 1.0, {"task_id": task.id, "note": "no_expected_actions"}

    # If action.task_id is provided, only count actions for this task; otherwise count as task-agnostic.
    filtered: List[Action] = [a for a in history if a.task_id in (None, task.id)]

    matched = 0
    expected_idx = 0
    wrong_or_extra = 0
    out_of_order_hits = 0

    for a in filtered:
        if expected_idx >= len(expected):
            wrong_or_extra += 1
            continue
        exp = expected[expected_idx]
        if _action_matches_expected(a, exp.type, exp.content_contains):
            matched += 1
            expected_idx += 1
        else:
            # Out-of-order: matches a later expected step
            for j in range(expected_idx + 1, len(expected)):
                if _action_matches_expected(a, expected[j].type, expected[j].content_contains):
                    out_of_order_hits += 1
                    break
            wrong_or_extra += 1

    correctness = matched / float(len(expected))
    completion = 1.0 if matched == len(expected) else 0.0

    optimal_steps = float(len(expected))
    used_steps = float(max(1, len(filtered)))
    efficiency = max(0.0, min(1.0, optimal_steps / used_steps))

    importance = _importance_multiplier(task)
    # Priority handling: prefer fewer out-of-order and wrong steps on important tasks.
    denom = float(max(1, len(filtered)))
    wrong_ratio = wrong_or_extra / denom
    out_of_order_ratio = out_of_order_hits / denom
    priority_handling = max(0.0, 1.0 - (wrong_ratio * 0.8 + out_of_order_ratio * 0.4))
    priority_handling = max(0.0, min(1.0, priority_handling * min(1.0, importance / 1.0)))

    # Combine into manager-style final score.
    score = (
        0.4 * correctness
        + 0.2 * completion
        + 0.2 * efficiency
        + 0.2 * priority_handling
    )

    # Apply explicit penalties.
    rep_pen = _repeat_penalty(filtered)
    noise_pen = _unnecessary_action_penalty(task, filtered)
    score = max(0.0, min(1.0, score - rep_pen - noise_pen))

    details: Dict[str, Any] = {
        "task_id": task.id,
        "expected_len": len(expected),
        "matched_in_order": matched,
        "wrong_or_extra": wrong_or_extra,
        "out_of_order_matches": out_of_order_hits,
        "components": {
            "correctness": correctness,
            "completion": completion,
            "efficiency": efficiency,
            "priority_handling": priority_handling,
        },
        "penalties": {"repeat_penalty": rep_pen, "unnecessary_action_penalty": noise_pen},
        "score": score,
    }
    return score, details


def grade_episode(*, task_id: str, history: List[Action]) -> Tuple[float, Dict[str, Any]]:
    """
    Grade an episode transcript against a known task id.
    """

    task = get_task(task_id)
    return grade_task(task, history)


def grade_state(
    state: State, *, task_id: Optional[str] = None, history: Optional[List[Action]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Grade from full environment state, optionally overriding task_id/history.
    """

    effective_task_id = task_id or state.current_task_id
    effective_history = history or state.history
    return grade_episode(task_id=effective_task_id, history=effective_history)
