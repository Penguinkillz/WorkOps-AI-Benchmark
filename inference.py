#!/usr/bin/env python3
"""
inference.py - OpenEnv Submission Inference Script for ai_workops_env

Mandatory env vars (set by evaluator):
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Optional env vars:
    LOCAL_IMAGE_NAME   Docker image to start (if using from_docker_image pattern).
    ENV_URL            Override environment base URL (default: http://localhost:7860).
"""

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Mandatory env vars per hackathon spec
# Defaults set ONLY for API_BASE_URL and MODEL_NAME (not HF_TOKEN)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    HF_TOKEN = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")

# ---------------------------------------------------------------------------
# Environment connection
# ---------------------------------------------------------------------------
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK = "ai_workops_env"
MAX_STEPS = 16
TEMPERATURE = 0.3
MAX_TOKENS = 512
SUCCESS_THRESHOLD = 0.1

VALID_ACTION_TYPES = [
    "reply", "ignore", "escalate", "resolve",
    "refund", "check_system", "file_bug",
]

SYSTEM_PROMPT = (
    "You are a workplace operations agent. You process items in a work inbox "
    "by taking actions one at a time.\n\n"
    "Valid action types: " + ", ".join(VALID_ACTION_TYPES) + "\n\n"
    "General guidance:\n"
    "- Read each inbox item carefully and pick the most appropriate action\n"
    "- Be concise in your responses\n"
    "- Handle items professionally\n\n"
    'Respond with a single JSON object: '
    '{"type": "<action_type>", "task_id": "<task_id>", "content": "<brief text>"}\n'
    "Return ONLY the JSON object, nothing else."
)


# ---------------------------------------------------------------------------
# Structured stdout logging (mandatory format)
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str] = None
) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Docker helper (optional, only if LOCAL_IMAGE_NAME is set)
# ---------------------------------------------------------------------------
_docker_proc: Optional[subprocess.Popen] = None


def start_docker_env() -> Optional[subprocess.Popen]:
    if not LOCAL_IMAGE_NAME:
        return None
    try:
        print(f"[DEBUG] Starting Docker container: {LOCAL_IMAGE_NAME}", file=sys.stderr, flush=True)
        proc = subprocess.Popen(
            ["docker", "run", "--rm", "-p", "7860:7860", LOCAL_IMAGE_NAME],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc
    except Exception as exc:
        print(f"[DEBUG] Docker start failed: {exc}", file=sys.stderr, flush=True)
        return None


def stop_docker_env(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def wait_for_env(timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{ENV_URL}/", timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------
def env_get(path: str) -> Dict[str, Any]:
    resp = requests.get(f"{ENV_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_post(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}{path}", json=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
def _extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def get_llm_action(
    client: Optional[OpenAI],
    obs: Dict[str, Any],
    task_id: str,
    past_actions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if client is None:
        return None

    obs_payload = {
        "step": obs.get("step", 0),
        "current_task_id": obs.get("current_task_id", task_id),
        "message": obs.get("message", ""),
        "inbox": obs.get("inbox", []),
    }

    history_block = ""
    if past_actions:
        hlines = []
        for i, act in enumerate(past_actions):
            hlines.append(f"Step {i}: {{type: {act['type']}, content: {act.get('content', '')}}}")
        history_block = (
            "\n\nActions already taken:\n"
            + "\n".join(hlines)
            + "\nConsider what you have already done before choosing your next action."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Current observation:\n"
                + json.dumps(obs_payload, ensure_ascii=True)
                + history_block
                + "\n\nPick the single best next action. Return strictly valid JSON only."
            ),
        },
    ]

    last_raw = ""
    for _attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            last_raw = (response.choices[0].message.content or "").strip()
            payload = json.loads(_extract_json(last_raw))
            action_type = payload.get("type", "reply")
            if action_type not in VALID_ACTION_TYPES:
                action_type = "reply"
            return {
                "type": action_type,
                "task_id": task_id,
                "content": str(payload.get("content", "")),
            }
        except Exception as exc:
            print(f"[DEBUG] LLM attempt failed: {exc}", file=sys.stderr, flush=True)
            messages.append({"role": "assistant", "content": last_raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Invalid JSON. Return ONLY a JSON object like: "
                        + '{"type": "reply", "task_id": "' + task_id + '", "content": "your message"}'
                    ),
                }
            )

    return None


def heuristic_fallback(task_id: str) -> Dict[str, Any]:
    return {
        "type": "reply",
        "task_id": task_id,
        "content": "Acknowledged, looking into this.",
    }


def format_action_str(action: Dict[str, Any]) -> str:
    content = (action.get("content") or "")[:60].replace("\n", " ")
    return f"{action['type']}(\"{content}\")" 


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    docker_proc = start_docker_env()

    try:
        if not wait_for_env(timeout=60 if docker_proc else 15):
            print(f"[ERROR] Environment not reachable at {ENV_URL}", file=sys.stderr)
            return

        client: Optional[OpenAI] = None
        if not HF_TOKEN:
            print("[DEBUG] HF_TOKEN missing; using heuristic fallback only.", file=sys.stderr, flush=True)
        else:
            try:
                client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
            except Exception as exc:
                print(
                    f"[DEBUG] OpenAI client init failed ({exc}); using heuristic fallback only.",
                    file=sys.stderr,
                    flush=True,
                )
                client = None

        try:
            tasks_resp = env_get("/tasks")
            tasks = tasks_resp.get("tasks", [])
        except Exception as exc:
            print(f"[DEBUG] /tasks fetch failed: {exc}", file=sys.stderr, flush=True)
            tasks = []

        if not tasks:
            tasks = [
                {"id": "easy_email_triage"},
                {"id": "medium_support_resolution"},
                {"id": "hard_workflow_refund_bug_escalation"},
            ]

        for task in tasks:
            task_id = task["id"]
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            try:
                obs = env_post("/reset", {"task_id": task_id})
                done = False
                past_actions: List[Dict[str, Any]] = []

                for step in range(1, MAX_STEPS + 1):
                    if done:
                        break

                    action = get_llm_action(client, obs, task_id, past_actions)
                    if action is None:
                        action = heuristic_fallback(task_id)

                    result = env_post("/step", action)
                    reward = float(result.get("reward", {}).get("value", 0.0))
                    done = bool(result.get("done", False))
                    error_msg = result.get("info", {}).get("error")

                    rewards.append(reward)
                    steps_taken = step
                    past_actions.append(action)
                    obs = result.get("observation", {})

                    log_step(
                        step=step,
                        action=format_action_str(action),
                        reward=reward,
                        done=done,
                        error=error_msg,
                    )

                grade = env_post("/grader", {"task_id": task_id})
                score = float(grade.get("score", 0.0))
                score = min(max(score, 0.0), 1.0)
                success = score >= SUCCESS_THRESHOLD

            except Exception as exc:
                print(
                    f"[DEBUG] Task {task_id} error: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

            finally:
                log_end(
                    success=success,
                    steps=steps_taken,
                    score=score,
                    rewards=rewards,
                )

    finally:
        stop_docker_env(docker_proc)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] Unhandled inference exception: {exc}", file=sys.stderr, flush=True)
