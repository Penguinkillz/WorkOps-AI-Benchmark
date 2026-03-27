# AI WorkOps Environment — Real-World AI Agent Evaluation Benchmark

## Overview
This project is an OpenEnv-compliant environment that simulates real business workflows for agent evaluation.

Agents act like operations managers handling operational work items such as:
email triage, customer support resolutions, and multi-step internal workflows.

It is designed to evaluate AI agents on real-world decision-making: what to do, in what order, how quickly, and how to handle priority work correctly.

## Key Features
- Realistic task simulation (email triage, support, workflows)
- Multi-step reasoning tasks with sequence-awareness
- Hidden priority signals (VIP, urgency, risk) used for evaluation
- Multi-factor reward system (correctness, efficiency, priority awareness, action quality)
- Deterministic grading with a final score in the range `0..1`
- OpenEnv-compliant API (FastAPI + Pydantic schemas)
- Baseline agent included (a “decent but imperfect employee”)

## Environment Design
The environment follows a standard OpenEnv loop:

- **Observation space**: the agent sees remaining inbox items plus the last action taken.
- **Action space**: operations manager actions (e.g., `reply`, `ignore`, `escalate`, `refund`, `file_bug`, `check_system`, `resolve`), with optional content.
- **State transitions**: each `step()` validates and applies the action to the correct task, updating internal progress and inbox state.
- **Reward logic (high level)**: reward is shaped by correctness + order/sequence, reduced by time and poor behavior, and scaled by hidden VIP/urgency/risk importance.

## Tasks
- **Easy: Email triage**
  A batch of realistic emails (refund request, spam, urgent bug report, VIP message). The agent must triage using `reply`, `ignore`, or `escalate`.

- **Medium: Support resolution**
  A multi-message support conversation with explicit tone and contextual fields (order id, issue type). The agent must `reply` with a proper resolution and then `resolve`, escalating when needed.

- **Hard: Multi-step workflow**
  A complex operational case involving conflicting internal system status and hidden VIP context. The agent must follow a multi-step sequence: check internal systems, act on the refund, file the bug, escalate to engineering, respond, and resolve.

## Evaluation
Scoring is deterministic and intended to reflect a manager-style review of agent performance.

- **Grader components**
  - Completion score: did the agent finish the expected action sequence?
  - Correctness score: were actions aligned to the expected plan and order?
  - Efficiency score: fewer steps (relative to the optimal plan) earns higher credit.
  - Priority handling score: correct treatment of VIP/urgent/high-risk work increases the score.

- **Final score**
  The final score is normalized to `0..1` using a weighted combination:
  `0.4 * correctness + 0.2 * completion + 0.2 * efficiency + 0.2 * priority_handling`

- **Why the baseline is imperfect**
  The included baseline agent is deliberately “good but not perfect”: it makes controlled mistakes such as occasional suboptimal ordering, generic responses, and small efficiency losses. This creates meaningful evaluation signal for stronger agents.

## API Endpoints
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /baseline`

## Running Locally
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Start the server:
   - `uvicorn app.main:app --reload`

The API runs on `http://127.0.0.1:8000` by default.

## Docker
Build:
- `docker build -t ai-workops-env .`

Run:
- `docker run -p 7860:7860 ai-workops-env`

Inside Docker, the service runs on `http://127.0.0.1:7860`.

## Hugging Face Deployment
To deploy on Hugging Face Spaces, set the container to listen on port `7860` (already configured in the Dockerfile).

Typical workflow:
- Build the Docker image locally (optional).
- Push the code to a Space repository.
- Configure the Space to use the Dockerfile build.

Space link: `https://huggingface.co/spaces/<your-username>/<your-space-name>`

## Why This Matters
This is not a toy benchmark.

It evaluates practical agent abilities that matter in real operations:
sequencing, prioritization, efficient task handling, and behavior under ambiguity.

The deterministic 0..1 scoring and OpenEnv-compatible API make it useful for RL training loops and agent benchmarking in hackathon and competition settings.
