from __future__ import annotations

from typing import Any, Dict, List

from app.env import Environment
from app.grader import grade_episode
from app.models import Action
from app.tasks import ExpectedAction, list_tasks


def _baseline_action_for_expected(expected: ExpectedAction, *, quality: str = "good") -> Action:
    """
    Deterministic baseline policy action for an expected step.

    quality:
    - "good": includes required content when present
    - "generic": likely misses content requirements
    """

    t = expected.type
    required = (expected.content_contains or "").strip()

    def with_required(base: str) -> str:
        if quality == "generic":
            return base
        if not required:
            return base
        # Ensure required substring appears verbatim (case-insensitive checks).
        if required.lower() in base.lower():
            return base
        return f"{base} ({required})"

    if t == "reply":
        return Action(type="reply", content=with_required("Thanks, I reviewed this and will follow up shortly."))
    if t == "file_bug":
        return Action(type="file_bug", content=with_required("Created bug report for this issue."))
    if t == "escalate":
        return Action(type="escalate", content=with_required("Escalating internally with context."))
    if t == "check_system":
        return Action(type="check_system", content=with_required("Checked internal systems for status."))
    if t == "refund":
        return Action(type="refund", content=with_required("Refund processed according to policy."))
    if t == "resolve":
        return Action(type="resolve", content=with_required("Resolved and documented outcome."))
    if t == "ignore":
        return Action(type="ignore", content=None)
    return Action(type=t, content=with_required(""))


def run_baseline(*, max_steps: int = 8) -> Dict[str, Any]:
    """
    Run a deterministic "decent but imperfect" baseline on each task.
    It intentionally makes small, controlled mistakes while still completing tasks.
    """

    results: List[Dict[str, Any]] = []
    for task in list_tasks():
        env = Environment(max_steps=max_steps)
        env.reset(task_id=task.id)

        done = False
        inserted_noise = 0
        while not done:
            # Always take the next expected action for this task.
            expected_idx = env.state().info.get("progress", {}).get(task.id, {}).get("progress_idx", 0)
            expected_idx = int(expected_idx)

            # Guard if task already complete.
            if expected_idx >= len(task.expected):
                break

            expected = task.expected[expected_idx]

            # Deterministic imperfections by difficulty.
            if task.difficulty == "easy":
                # One unnecessary action before first correct step (small efficiency hit).
                if expected_idx == 0 and inserted_noise == 0 and env.current_step < max_steps - 1:
                    action = Action(type="reply", task_id=task.id, content="Thanks, we will check this.")
                    inserted_noise += 1
                else:
                    action = _baseline_action_for_expected(expected, quality="good")
                    action.task_id = task.id

            elif task.difficulty == "medium":
                # First attempt is generic reply (may miss required keyword), then correct sequence.
                if expected_idx == 0 and inserted_noise == 0 and env.current_step < max_steps - 1:
                    action = _baseline_action_for_expected(expected, quality="generic")
                    action.task_id = task.id
                    inserted_noise += 1
                # Slightly suboptimal: one extra status-check style reply after step 2 if room.
                elif env.current_step >= 2 and inserted_noise == 1 and env.current_step < max_steps - 1:
                    action = Action(type="reply", task_id=task.id, content="Quick update: still investigating.")
                    inserted_noise += 1
                else:
                    action = _baseline_action_for_expected(expected, quality="good")
                    action.task_id = task.id

            else:  # hard
                # Controlled degradation:
                # - do a few deterministic out-of-order/noisy actions
                # - still complete most of the workflow without crashing
                if expected_idx == 0 and inserted_noise == 0 and env.current_step < max_steps - 1:
                    action = Action(type="escalate", task_id=task.id, content="Escalating this issue.")
                    inserted_noise += 1
                elif expected_idx == 2 and inserted_noise == 1 and env.current_step < max_steps - 1:
                    action = Action(type="reply", task_id=task.id, content="We are looking into this.")
                    inserted_noise += 1
                elif expected_idx == 4 and inserted_noise == 2 and env.current_step < max_steps - 1:
                    action = Action(type="resolve", task_id=task.id, content="Marking this as resolved.")
                    inserted_noise += 1
                elif expected.type in ("file_bug", "escalate", "reply") and env.current_step > 2:
                    # Keep required content for progress, but writing remains generic-ish.
                    action = _baseline_action_for_expected(expected, quality="good")
                    action.task_id = task.id
                else:
                    action = _baseline_action_for_expected(expected, quality="good")
                    action.task_id = task.id

            step_res = env.step(action)
            done = step_res.done

            # Safety: avoid infinite loops if something unexpected occurs.
            if env.current_step >= max_steps:
                break

        score, details = grade_episode(task_id=task.id, history=env.history)
        results.append(
            {
                "task_id": task.id,
                "difficulty": task.difficulty,
                "score": score,
                "details": details,
                "steps": env.current_step,
            }
        )

    avg = sum(r["score"] for r in results) / float(max(1, len(results)))
    return {"results": results, "average_score": avg}
