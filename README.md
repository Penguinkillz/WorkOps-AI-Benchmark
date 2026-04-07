---
title: WorkOps AI Benchmark
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# WorkOps AI Benchmark

A production-style OpenEnv environment for evaluating LLM agents on real operations workflows: inbox triage, support resolution, and incident escalation.

## Why this project exists

Most benchmarks test abstract games or toy tasks. Real teams need agents that can handle operational work under ambiguity and time pressure.

This environment evaluates whether an agent can:
- pick correct actions from realistic business context,
- follow multi-step procedures in order,
- avoid noisy or repetitive behavior,
- resolve tasks with measurable quality.

## What is modeled

Three deterministic tasks with increasing difficulty:

- **Easy - `easy_email_triage`**
  - Triage a queue of real-looking inbound emails using `reply`, `ignore`, `escalate`.
- **Medium - `medium_support_resolution`**
  - Resolve support scenarios with variant-specific policy paths.
- **Hard - `hard_workflow_refund_bug_escalation`**
  - Handle a cross-functional workflow: system checks, refund, bug filing, escalation, customer response, closure.

## API (OpenEnv style)

The service exposes typed interfaces over HTTP:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /baseline`
- `GET /` (health)

Schemas are implemented with Pydantic models in `app/models.py` and aligned with `openenv.yaml`.

## Reward and grading design

- Step reward is deterministic and normalized to `[0.0, 1.0]`.
- Reward shaping includes correctness, ordering, timing/efficiency, and noise penalties.
- Episode grading (`/grader`) is deterministic and returns final score in `[0.0, 1.0]`.

This gives meaningful partial-credit signals instead of binary pass/fail.

## Local run

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Health check:

```bash
curl http://127.0.0.1:7860/
```

## Docker run

```bash
docker build -t ai-workops-env .
docker run -p 7860:7860 ai-workops-env
```

## OpenEnv validation

```bash
openenv validate
```

## Baselines

### 1) LLM baseline (`GET /baseline`)
Uses OpenAI-compatible client against Groq.

Environment variables:
- required: `GROQ_API_KEY`
- optional: `LLM_BASE_URL` (default `https://api.groq.com/openai/v1`)
- optional: `LLM_MODEL` (default `llama-3.3-70b-versatile`)

### 2) Heuristic baseline
`run_heuristic_baseline()` is deterministic and useful for local regression checks without external credentials.

Sample deterministic scores:

| Task | Score |
|---|---:|
| easy_email_triage | 0.8420 |
| medium_support_resolution | 0.7160 |
| hard_workflow_refund_bug_escalation | 0.7511 |
| **Average** | **0.7697** |

## Submission-relevant files

- `openenv.yaml` - environment endpoint metadata
- `Dockerfile` - deploy/runtime image
- `inference.py` - evaluator-facing inference script with required START/STEP/END logs
- `pyproject.toml` - package metadata used by validation

## Repository structure

- `app/env.py` - environment transition + reward logic
- `app/tasks.py` - task definitions and expected trajectories
- `app/grader.py` - deterministic episode scoring
- `app/main.py` - FastAPI routes
- `app/baseline.py` - baseline agents
- `server/app.py` - `server` entry point for OpenEnv tooling