---
title: WorkOps AI Benchmark
emoji: "??"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# AI WorkOps Environment

`ai_workops_env` is a production-style OpenEnv benchmark for evaluating AI agents on realistic operations workflows: inbox triage, support resolution, and cross-functional incident handling.

## Problem Statement

Most agent environments test game mechanics or synthetic tasks. Real organizations need agents that can:
- interpret ambiguous customer and internal messages,
- choose the right operational action sequence under time pressure,
- avoid noisy or unsafe actions, and
- resolve work with measurable quality.

This project models that gap with deterministic tasks, typed APIs, and reproducible grading.

## Why This Environment Exists

The benchmark is designed for practical agent evaluation in business operations contexts:
- **Real-world utility:** tasks resemble daily support/ops workflows.
- **Structured evaluation:** clear expected trajectories and deterministic scoring.
- **Partial-credit learning signal:** reward shaping captures progress, timing, and quality.
- **OpenEnv compatibility:** standard `reset()` / `step()` / `state()` interface for agent runners.

## Environment Overview

### Task Set
- `easy_email_triage` (easy): process an email queue with `reply`, `ignore`, and `escalate`.
- `medium_support_resolution` (medium): handle support variants with scenario-aware expectations.
- `hard_workflow_refund_bug_escalation` (hard): execute a multi-step workflow across checks, refunding, bug filing, escalation, and closure.

### Episode Boundaries
- easy: 10 max steps
- medium: 12 max steps
- hard: 16 max steps

Episodes end when expected steps are completed or max steps are reached.

## API Contract (OpenEnv-style)

### Action
`Action` is a JSON object with:
- `type` (required string): e.g. `reply`, `ignore`, `escalate`, `check_system`, `refund`, `file_bug`, `resolve`
- `task_id` (optional string): defaults to `current_task_id`
- `content` (optional string): response or rationale text
- `metadata` (optional object): extra context

### Observation
`Observation` returns:
- `step` (int)
- `current_task_id` (string)
- `inbox` (array of `InboxItem`)
- `last_action` (optional `Action`)
- `message` (string)

`InboxItem` includes:
- `id`, `kind`, `subject`, `body`, `difficulty`, `metadata`

## Reward and Grading

- Step-level reward is deterministic and exposed as `Reward.value` in `[0.0, 1.0]`.
- Internal components can be negative before clamp, then normalized to `[0.0, 1.0]`.
- Final episode score via `POST /grader` is deterministic in `[0.0, 1.0]`.

The reward function includes correctness, ordering, efficiency, and penalties for repetition/noise.

## Endpoints

- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /baseline`

Endpoint definitions align with `openenv.yaml`.

## Project Structure

- `app/main.py`: FastAPI entrypoint and routes
- `app/env.py`: environment dynamics and reward logic
- `app/tasks.py`: canonical tasks and expected flows
- `app/grader.py`: deterministic grading logic
- `app/models.py`: typed action/observation/state/reward schemas
- `app/baseline.py`: baseline runners (LLM + heuristic)
- `inference.py`: submission inference script with required stdout protocol
- `openenv.yaml`: endpoint/spec metadata
- `Dockerfile`: containerized runtime for local/HF deployment

## Local Development

### Requirements
- Python 3.10+
- pip

### Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Health check:
```bash
curl http://127.0.0.1:7860/
```

## Docker

```bash
docker build -t ai-workops-env .
docker run -p 7860:7860 ai-workops-env
```

## Validation

```bash
openenv validate
```

Expected: validation passes with deployment-ready status.

## Baselines

### LLM baseline (`GET /baseline`)
Uses OpenAI-compatible client with Groq endpoint.

Env vars:
- required: `GROQ_API_KEY`
- optional: `LLM_BASE_URL` (default `https://api.groq.com/openai/v1`)
- optional: `LLM_MODEL` (default `llama-3.3-70b-versatile`)

If the key is missing, endpoint returns HTTP `503` with `llm_baseline_unavailable`.

### Heuristic baseline
`run_heuristic_baseline()` provides deterministic local regression checks without credentials.

Sample heuristic scores:

| Task | Score |
|---|---:|
| easy_email_triage | 0.8420 |
| medium_support_resolution | 0.7160 |
| hard_workflow_refund_bug_escalation | 0.7511 |
| **Average** | **0.7697** |

## Submission Notes

- `inference.py` is placed at repo root and follows required `[START]`, `[STEP]`, `[END]` stdout format.
- API-facing scores are normalized to `[0,1]`.
- Docker build and OpenEnv validation are part of pre-submission checks.

## License

For hackathon evaluation use. Add your preferred license if publishing publicly beyond the competition.
