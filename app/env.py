from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.models import Action, EnvMetadata, InboxItem, Observation, Reward, State, StepResult
from app.tasks import TaskDefinition, get_task, list_tasks, pick_task_by_difficulty


@dataclass
class _TaskRuntime:
    definition: TaskDefinition
    progress_idx: int = 0
    handled: bool = False


class Environment:
    """
    Deterministic in-memory environment for "ai_workops_env".

    The environment maintains a small inbox of work items (tasks). Agents act by
    taking actions (e.g., reply/escalate/resolve). Correctness is determined by
    comparing the agent's actions to each task's expected action sequence.
    """

    def __init__(self, *, env_name: str = "ai_workops_env", max_steps: int = 8) -> None:
        self.env_name = env_name
        self.max_steps = max_steps
        self._episode_id: str = ""
        self._history: List[Action] = []
        self._current_step: int = 0
        self._task_runtimes: Dict[str, _TaskRuntime] = {}
        self._inbox: List[InboxItem] = []
        self._done: bool = False
        self._info: Dict[str, Any] = {}

    @property
    def inbox(self) -> List[InboxItem]:
        return list(self._inbox)

    @property
    def history(self) -> List[Action]:
        return list(self._history)

    @property
    def current_step(self) -> int:
        return self._current_step

    def reset(self, *, task_id: Optional[str] = None, difficulty: Optional[str] = None) -> Observation:
        """
        Reset the environment to a fresh episode.

        If `task_id` is provided, load only that task.
        If `difficulty` is provided, load only a task of that difficulty.
        Otherwise, load the canonical task set (easy/medium/hard) into the inbox.
        """

        self._episode_id = str(uuid.uuid4())
        self._history = []
        self._current_step = 0
        self._done = False
        self._info = {"progress": {}}

        tasks: List[TaskDefinition]
        if task_id is not None:
            tasks = [get_task(task_id)]
        elif difficulty is not None:
            tasks = [pick_task_by_difficulty(difficulty)]  # type: ignore[arg-type]
        else:
            tasks = list_tasks()
            # Deterministic "variability": shuffle task order with a fixed seed.
            rng = random.Random(1337)
            rng.shuffle(tasks)

        self._task_runtimes = {t.id: _TaskRuntime(definition=t) for t in tasks}
        self._inbox = [self._to_inbox_item(t) for t in tasks]
        self._sync_progress_info()

        return self._make_observation(last_action=None, message="Environment reset.")

    def step(self, action: Action) -> StepResult:
        """
        Apply an agent action, update state, and return observation + reward.
        """

        if self._done:
            obs = self._make_observation(last_action=action, message="Episode already done. Call reset().")
            return StepResult(
                observation=obs,
                reward=Reward(value=0.0, components={"terminal": 0.0}),
                done=True,
                info={"error": "episode_done"},
            )

        self._validate_action(action)

        target_task_id = self._resolve_task_id(action)
        runtime = self._task_runtimes[target_task_id]

        reward_value, reward_components, applied, reward_debug = self._apply_action(runtime, action)
        self._history.append(action)
        self._current_step += 1

        # Termination conditions
        if self._all_tasks_handled() or self._current_step >= self.max_steps:
            self._done = True

        self._sync_progress_info()

        message = "Action applied." if applied else "Action recorded (no effect)."
        if self._done:
            message = "Episode done."

        obs = self._make_observation(last_action=action, message=message)
        info: Dict[str, Any] = {
            "episode_id": self._episode_id,
            "task_id": target_task_id,
            "applied": applied,
            "progress": self._info.get("progress", {}),
            "reward_debug": reward_debug,
        }
        if self._current_step >= self.max_steps and not self._all_tasks_handled():
            info["termination_reason"] = "max_steps"
        elif self._all_tasks_handled():
            info["termination_reason"] = "all_tasks_handled"

        # Clamp reward to [0, 1]
        reward_value = max(0.0, min(1.0, reward_value))
        return StepResult(
            observation=obs,
            reward=Reward(value=reward_value, components=reward_components),
            done=self._done,
            info=info,
        )

    def state(self) -> State:
        """
        Return the full internal state (OpenEnv-friendly JSON).
        """

        current_task_id = self._current_task_id()
        return State(
            env_name=self.env_name,
            episode_id=self._episode_id,
            current_task_id=current_task_id,
            inbox=self.inbox,
            history=self.history,
            metadata=EnvMetadata(timestep=self._current_step, max_steps=self.max_steps),
            done=self._done,
            info=dict(self._info),
        )

    # -----------------------
    # Internals
    # -----------------------

    def _to_inbox_item(self, task: TaskDefinition) -> InboxItem:
        subj = task.input.get("email", {}).get("subject") or task.input.get("ticket", {}).get("subject") or task.title
        body = (
            task.input.get("email", {}).get("body")
            or task.input.get("ticket", {}).get("body")
            or task.input.get("case", {}).get("issue")
            or task.input.get("case", {}).get("issue_summary")
            or task.description
        )
        kind = "workflow" if task.difficulty == "hard" else ("ticket" if task.difficulty == "medium" else "email")
        visible_metadata = {k: v for k, v in dict(task.metadata).items() if k != "hidden"}
        return InboxItem(
            id=task.id,
            kind=kind,  # type: ignore[arg-type]
            subject=str(subj),
            body=str(body),
            difficulty=task.difficulty,
            metadata=visible_metadata,
        )

    def _current_task_id(self) -> str:
        if self._inbox:
            return self._inbox[0].id
        # If inbox empty, fall back to any task id for consistent schema
        if self._task_runtimes:
            return next(iter(self._task_runtimes.keys()))
        return ""

    def _resolve_task_id(self, action: Action) -> str:
        if action.task_id:
            if action.task_id not in self._task_runtimes:
                raise ValueError(f"Unknown task_id: {action.task_id}")
            return action.task_id
        return self._current_task_id()

    def _validate_action(self, action: Action) -> None:
        if not isinstance(action.type, str) or not action.type.strip():
            raise ValueError("Action.type must be a non-empty string")
        if action.task_id is not None and not isinstance(action.task_id, str):
            raise ValueError("Action.task_id must be a string when provided")
        if action.content is not None and not isinstance(action.content, str):
            raise ValueError("Action.content must be a string when provided")
        if action.metadata is None or not isinstance(action.metadata, dict):
            raise ValueError("Action.metadata must be an object")

    def _apply_action(self, runtime: _TaskRuntime, action: Action) -> Tuple[float, Dict[str, float], bool]:
        """
        Compare `action` against the next expected action for the task.

        Manager-style reward design (deterministic):
        - multi-factor: correctness + order + efficiency + priority awareness + quality/timing
        - time penalty each step
        - harsher penalties for VIP/high-urgency/high-risk mistakes
        - bonuses for correctly resolving high-risk work
        """

        hidden = dict(runtime.definition.metadata.get("hidden", {})) if isinstance(runtime.definition.metadata, dict) else {}
        vip = bool(hidden.get("vip_flag", False))
        urgency = str(hidden.get("urgency", "normal")).lower()
        risk = str(hidden.get("risk_level", "medium")).lower()

        importance = self._importance_multiplier(vip=vip, urgency=urgency, risk=risk)

        if runtime.handled:
            return 0.0, {"already_handled": 0.0}, False, {"reason": "already_handled"}

        expected_list = runtime.definition.expected
        if runtime.progress_idx >= len(expected_list):
            runtime.handled = True
            self._remove_from_inbox(runtime.definition.id)
            return 0.0, {"task_already_complete": 0.0}, False, {"reason": "task_already_complete"}

        # Determine match quality against the expected sequence.
        match = self._sequence_match(runtime=runtime, action=action)

        components: Dict[str, float] = {}
        debug: Dict[str, Any] = {
            "expected_next": expected_list[runtime.progress_idx].model_dump(),
            "importance_multiplier": importance,
            "vip": vip,
            "urgency": urgency,
            "risk": risk,
            "match": match,
        }

        # Bad behavior penalties (noise/repetition).
        noise_penalty = self._noise_penalty(action=action)
        if noise_penalty != 0.0:
            components["noise_penalty"] = noise_penalty

        # Time penalty encourages fast resolution (applies every step).
        time_penalty = -0.03 * importance
        components["time_penalty"] = time_penalty

        # Urgency delay penalty if urgent/high-risk work is still pending.
        delay_penalty = self._delay_penalty_for_outstanding_work()
        if delay_penalty != 0.0:
            components["delay_penalty"] = delay_penalty

        # Main reward based on match quality.
        delta = 0.0
        applied = True

        if match["kind"] == "correct_next":
            base = (0.45 + (0.10 if match["content_ok"] else 0.0)) * importance
            components["correct_next"] = base
            delta += base
            runtime.progress_idx += 1
        elif match["kind"] == "correct_but_out_of_order":
            # Good intent but poor timing/order.
            partial = (0.20 + (0.05 if match["content_ok"] else 0.0)) * importance
            components["out_of_order_partial"] = partial
            delta += partial
        elif match["kind"] == "correct_type_missing_content":
            partial = 0.15 * importance
            components["partial_missing_content"] = partial
            delta += partial
        else:
            # Wrong action: scale penalty by severity and importance. VIP mistakes are harsher.
            severity = float(match.get("severity", 1.0))
            wrong = -min(1.0, 0.35 * severity * importance)
            if vip and urgency in ("high", "urgent"):
                wrong = min(wrong, -0.60)
            components["wrong_action"] = wrong
            delta += wrong

        # Completion bonus and risk bonus.
        completed = runtime.progress_idx >= len(expected_list)
        if completed:
            runtime.handled = True
            self._remove_from_inbox(runtime.definition.id)
            complete_bonus = 0.25 * importance
            components["task_complete_bonus"] = complete_bonus
            delta += complete_bonus

            if risk == "high":
                risk_bonus = 0.10
                components["high_risk_resolution_bonus"] = risk_bonus
                delta += risk_bonus

            eff = self._efficiency_bonus()
            if eff > 0:
                components["efficiency_bonus"] = eff
                delta += eff

        delta += noise_penalty + time_penalty + delay_penalty

        # Clamp step reward into [-1, 1] before the outer [0,1] clamp in step().
        delta = max(-1.0, min(1.0, delta))
        debug["components"] = dict(components)
        debug["delta_before_clamp"] = delta
        debug["progress_idx_after"] = runtime.progress_idx
        debug["task_completed"] = completed

        return delta, components, applied, debug

    def _importance_multiplier(self, *, vip: bool, urgency: str, risk: str) -> float:
        """
        Importance multiplier used to scale rewards/penalties for manager realism.
        """

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

    def _sequence_match(self, *, runtime: _TaskRuntime, action: Action) -> Dict[str, Any]:
        """
        Sequence-aware match against the task's expected action list.

        Returns a dict describing match kind:
        - correct_next
        - correct_but_out_of_order
        - correct_type_missing_content
        - wrong
        """

        expected_list = runtime.definition.expected
        next_expected = expected_list[runtime.progress_idx]

        def content_ok(exp_contains: Optional[str]) -> bool:
            if exp_contains is None:
                return True
            return exp_contains.lower() in (action.content or "").lower()

        # Exact next action match
        if action.type == next_expected.type:
            ok = content_ok(next_expected.content_contains)
            if ok:
                return {"kind": "correct_next", "content_ok": True}
            return {"kind": "correct_type_missing_content", "content_ok": False}

        # Out-of-order match: action matches a later expected step.
        for j in range(runtime.progress_idx + 1, len(expected_list)):
            exp = expected_list[j]
            if action.type == exp.type:
                return {"kind": "correct_but_out_of_order", "content_ok": content_ok(exp.content_contains), "expected_index": j}

        # Wrong action: estimate severity by whether it conflicts with policy-like steps.
        severity = 1.0
        if action.type in ("ignore",) and runtime.definition.difficulty in ("medium", "hard"):
            severity = 2.0
        if action.type in ("resolve",) and runtime.definition.difficulty == "hard" and runtime.progress_idx == 0:
            severity = 2.5
        return {"kind": "wrong", "severity": severity}

    def _noise_penalty(self, *, action: Action) -> float:
        """
        Penalize unnecessary/repeated behavior.
        """

        if not self._history:
            return 0.0

        last = self._history[-1]
        if last.type == action.type and (last.task_id == action.task_id):
            return -0.08
        return 0.0

    def _delay_penalty_for_outstanding_work(self) -> float:
        """
        Penalize delaying urgent VIP/high-risk items that remain in the inbox.
        This is based on hidden metadata, but does not expose it.
        """

        penalty = 0.0
        for rt in self._task_runtimes.values():
            if rt.handled:
                continue
            hidden = dict(rt.definition.metadata.get("hidden", {})) if isinstance(rt.definition.metadata, dict) else {}
            vip = bool(hidden.get("vip_flag", False))
            urgency = str(hidden.get("urgency", "normal")).lower()
            risk = str(hidden.get("risk_level", "medium")).lower()
            if vip or urgency in ("high", "urgent") or risk in ("high",):
                # Grows slowly with time (deterministic).
                penalty -= min(0.08, 0.01 * float(self._current_step))
        return penalty

    def _efficiency_bonus(self) -> float:
        """
        Bonus for finishing tasks in fewer steps overall.
        """

        # Max bonus if completing while still in first half of allowed steps.
        if self.max_steps <= 0:
            return 0.0
        ratio = self._current_step / float(self.max_steps)
        if ratio <= 0.25:
            return 0.15
        if ratio <= 0.5:
            return 0.10
        if ratio <= 0.75:
            return 0.05
        return 0.0

    def _remove_from_inbox(self, task_id: str) -> None:
        self._inbox = [i for i in self._inbox if i.id != task_id]

    def _all_tasks_handled(self) -> bool:
        return all(rt.handled for rt in self._task_runtimes.values()) if self._task_runtimes else True

    def _sync_progress_info(self) -> None:
        self._info["progress"] = {
            task_id: {
                "difficulty": rt.definition.difficulty,
                "progress_idx": rt.progress_idx,
                "expected_len": len(rt.definition.expected),
                "handled": rt.handled,
            }
            for task_id, rt in self._task_runtimes.items()
        }

    def _make_observation(self, *, last_action: Optional[Action], message: str) -> Observation:
        return Observation(
            step=self._current_step,
            current_task_id=self._current_task_id(),
            inbox=self.inbox,
            last_action=last_action,
            message=message,
        )
