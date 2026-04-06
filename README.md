# AI WorkOps Environment

## What This Is
`ai_workops_env` is an OpenEnv-style operations benchmark for evaluating AI agents on realistic workplace tasks (not game mechanics). It focuses on practical skills: inbox triage, support handling, and multi-step internal workflows with deterministic scoring.

## Why This Matters
Real-world operations work requires correct sequencing, prioritization, and safe execution under ambiguity. This environment measures those behaviors through constrained actions, structured observations, and reproducible grading.

## Action Space
Agent actions are JSON objects matching `Action`:
- `type` (string, required): action identifier, e.g. `reply`, `ignore`, `escalate`, `check_system`, `refund`, `file_bug`, `resolve`.
- `task_id` (string, optional): target task. If omitted, environment uses `current_task_id`.
- `content` (string, optional): free-form text used by content checks.
- `metadata` (object, optional): extra action fields.

## Observation Space
Each step returns `Observation`:
- `step` (int): current timestep.
- `current_task_id` (string): active task id.
- `inbox` (array of `InboxItem`): each item has:
  - `id`, `kind`, `subject`, `body`, `difficulty`, `metadata`.
- `last_action` (optional `Action`): previous action.
- `message` (string): environment status text.

Hidden policy metadata is used internally for reward/grading and is not exposed in observations.

## Tasks
- `easy_email_triage` (easy)
  - Objective: triage a queue of realistic emails in sequence using `reply`, `ignore`, `escalate`.
- `medium_support_resolution` (medium)
  - Objective: handle support conversation variants with variant-specific expected flows (login vs refund).
- `hard_workflow_refund_bug_escalation` (hard)
  - Objective: complete a six-step workflow across system check, refund, bug filing, escalation, customer reply, and resolve.

## Rewards and Grading
Environment step rewards are deterministic and exposed via `Reward.value` in the range `[0.0, 1.0]`.
Internal reward components may go negative before clamping; the API output is clamped to `[0.0, 1.0]`.

Episode grading (`/grader`) outputs a deterministic final score in `[0.0, 1.0]`.

## Max Steps and Termination
Per-episode step caps are difficulty-aware:
- easy: 10
- medium: 12
- hard: 16

Episodes terminate on either:
- all expected task steps handled, or
- max step limit reached.

## Setup (Local)
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Service URL: `http://127.0.0.1:7860`

## Docker
```bash
docker build -t ai-workops-env .
docker run -p 7860:7860 ai-workops-env
```

Health check:
```bash
curl http://127.0.0.1:7860/
```

## Baseline (Groq via OpenAI-Compatible Client)
The submission baseline is LLM-based and uses Groq with the OpenAI-compatible `openai` Python client.

Required env var:
- `GROQ_API_KEY`

Optional env vars:
- `LLM_BASE_URL` (default: `https://api.groq.com/openai/v1`)
- `LLM_MODEL` (default: `llama-3.3-70b-versatile`)

Local convenience:
- Put these values in `.env` (already supported by baseline loader) so you do not need to export env vars every run.
- Keep `.env` local only (it is git-ignored).

Baseline runtime settings:
- `temperature = 0.3`
- capped completion tokens per step
- max episode steps enforced by environment

### `/baseline` behavior
- If `GROQ_API_KEY` is missing: returns HTTP `503` with `llm_baseline_unavailable`.
- If key is present: runs LLM baseline and returns per-task scores + average.

### Heuristic Baseline (local debug only)
A deterministic heuristic baseline is retained in code as `run_heuristic_baseline()` for local testing and regression checks without API credentials.

### Baseline Scores

Heuristic baseline (deterministic, no API key needed):

| Task | Score |
|---|---:|
| easy_email_triage | 0.8420 |
| medium_support_resolution | 0.7160 |
| hard_workflow_refund_bug_escalation | 0.7511 |
| **Average** | **0.7697** |

LLM baseline (`GET /baseline`, requires `GROQ_API_KEY`): scores vary per run due to `temperature = 0.3`; typical average is **0.33 - 0.70** depending on the model and prompt alignment.

## Hugging Face Space
- Uses provided `Dockerfile` and port `7860`
- Add `openenv` tag if hackathon rules require it

## Endpoints
- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /baseline`

These match `openenv.yaml`.

## OpenEnv / Provider Note
Client library is OpenAI-compatible, provider is Groq. If your platform only accepts `OPENAI_API_KEY` secrets, follow organizer instructions for secret mapping rather than relabeling providers in docs.
