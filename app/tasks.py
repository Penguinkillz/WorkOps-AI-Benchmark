from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ExpectedAction(BaseModel):
    type: str
    content_contains: Optional[str] = Field(
        default=None, description="If set, action.content must contain this substring (case-insensitive)."
    )


class TaskDefinition(BaseModel):
    id: str
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    description: str
    input: Dict[str, Any]
    expected: List[ExpectedAction]
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class TaskCatalog:
    easy: TaskDefinition
    medium: TaskDefinition
    hard: TaskDefinition


def build_task_catalog() -> TaskCatalog:
    """
    Defines 3 canonical tasks for the environment.

    Keep tasks deterministic and self-contained for grading.
    """

    def pick(seed_key: str, variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        rng = random.Random(seed_key)
        return dict(rng.choice(variants))

    easy = TaskDefinition(
        id="easy_email_triage",
        difficulty="easy",
        title="Inbox triage (email)",
        description="Triage a small batch of real emails: reply / ignore / escalate.",
        input=pick(
            "easy_email_triage",
            [
                {
                    "queue": [
                        {
                            "email_id": "E-1001",
                            "sender": "refunds@buyer-mail.com",
                            "subject": "Refund for order ORD-21903?",
                            "body": "Hi team — I think I was billed twice. Order ORD-21903. Can you confirm and refund if needed?",
                            "visible_priority_hint": "normal",
                        },
                        {
                            "email_id": "E-1002",
                            "sender": "promo@superdeals.example",
                            "subject": "BOOST YOUR SALES (limited time)",
                            "body": "We can grow your business 10x. Reply YES to schedule a call.",
                            "visible_priority_hint": "low",
                        },
                        {
                            "email_id": "E-1003",
                            "sender": "dev@customerco.io",
                            "subject": "Checkout crash (logs attached)",
                            "body": "Our users are seeing a crash right after tapping Pay. Started this morning. Can you advise?",
                            "visible_priority_hint": "normal",
                        },
                        {
                            "email_id": "E-1004",
                            "sender": "alex.m@enterprise.example",
                            "subject": "Quick question about invoice terms",
                            "body": "Hey—can you confirm whether Net 30 applies to our renewal invoice? Thanks.",
                            "visible_priority_hint": "normal",
                        },
                    ]
                },
                {
                    # Slight wording variation, same underlying expected actions.
                    "queue": [
                        {
                            "email_id": "E-1101",
                            "sender": "customer@inboxmail.net",
                            "subject": "Need a refund — charged twice?",
                            "body": "Hello, I see two charges for the same order. Can you look into it and refund the duplicate?",
                            "visible_priority_hint": "normal",
                        },
                        {
                            "email_id": "E-1102",
                            "sender": "newsletters@randomsite.example",
                            "subject": "Your weekly digest",
                            "body": "Top tips to increase productivity. Unsubscribe any time.",
                            "visible_priority_hint": "low",
                        },
                        {
                            "email_id": "E-1103",
                            "sender": "support-eng@partner.io",
                            "subject": "URGENT: payment screen freezes",
                            "body": "Payment UI freezes for some users. We can reproduce. Please escalate to engineering.",
                            "visible_priority_hint": "normal",
                        },
                        {
                            "email_id": "E-1104",
                            "sender": "vip@enterprise.example",
                            "subject": "Invoice clarification",
                            "body": "Hi, small question about our invoice schedule. Appreciate a quick confirmation.",
                            "visible_priority_hint": "normal",
                        },
                    ]
                },
            ],
        ),
        expected=[
            # Deterministic order: handle in the queue order.
            ExpectedAction(type="reply", content_contains="order"),
            ExpectedAction(type="ignore"),
            ExpectedAction(type="escalate"),
            ExpectedAction(type="reply"),
        ],
        metadata={
            "domain": "inbox",
            "task_mode": "batch",
            "expected_policy": "Triage each email; escalate high-risk bugs; ignore spam; reply to legitimate requests.",
            "hidden": {
                "vip_sender_domains": ["enterprise.example"],
                "urgency_map": {"refund": "medium", "spam": "low", "bug": "high", "vip": "high"},
                "risk_level": "medium",
                # Hidden labels for grading/reward shaping (not exposed in observation).
                "labels": {
                    "refund": {"email_id": ["E-1001", "E-1101"], "vip": False, "urgency": "medium", "risk": "medium"},
                    "spam": {"email_id": ["E-1002", "E-1102"], "vip": False, "urgency": "low", "risk": "low"},
                    "bug": {"email_id": ["E-1003", "E-1103"], "vip": False, "urgency": "high", "risk": "high"},
                    "vip": {"email_id": ["E-1004", "E-1104"], "vip": True, "urgency": "high", "risk": "medium"},
                },
            },
        },
    )

    medium = TaskDefinition(
        id="medium_support_resolution",
        difficulty="medium",
        title="Support conversation: resolve or escalate",
        description="Handle a realistic support conversation with context, tone, and order details.",
        input=pick(
            "medium_support_resolution",
            [
                {
                    "ticket": {"id": "TCK-1042", "order_id": "ORD-55231", "issue_type": "login"},
                    "tone": "angry",
                    "conversation": [
                        {
                            "from": "customer",
                            "message": "I reset my password THREE times and your app still says 'invalid token'. This is ridiculous.",
                        },
                        {"from": "agent", "message": "Sorry about that. Can you share the exact error and device/browser?"},
                        {
                            "from": "customer",
                            "message": "Chrome on Windows. Error: invalid token. I need access today.",
                        },
                    ],
                    "internal_notes": {"known_issue": "stale browser cache after reset", "safe_fix": "clear cache + retry"},
                },
                {
                    "ticket": {"id": "TCK-1188", "order_id": "ORD-60019", "issue_type": "refund"},
                    "tone": "polite",
                    "conversation": [
                        {"from": "customer", "message": "Hi! I was billed twice for my subscription renewal (ORD-60019)."},
                        {"from": "customer", "message": "Could you please refund the duplicate charge? Thank you."},
                    ],
                    "internal_notes": {"known_issue": "duplicate renewal charge in rare cases", "requires": "escalate_billing"},
                },
            ],
        ),
        expected=[
            ExpectedAction(type="reply", content_contains="clear"),
            ExpectedAction(type="resolve"),
        ],
        metadata={
            "domain": "support",
            "expected_policy": "Give the correct resolution; escalate billing/system issues when required.",
            "hidden": {
                "vip_flag": False,
                "urgency": "medium",
                "risk_level": "medium",
                "escalate_required_if": ["refund_duplicate_charge"],
            },
        },
    )

    hard = TaskDefinition(
        id="hard_workflow_refund_bug_escalation",
        difficulty="hard",
        title="Multi-step workflow: conflicting payment + VIP + escalation",
        description="Handle a complex workplace workflow with conflicting internal status and hidden VIP context.",
        input=pick(
            "hard_workflow_refund_bug_escalation",
            [
                {
                    "case": {
                        "customer_id": "CUST-7781",
                        "order_id": "ORD-55231",
                        "issue_summary": "Customer reports payment failed but was charged; app crashed after checkout.",
                        "customer_message": "Your app crashed and I got charged. I want a refund today.",
                    },
                    "systems": {
                        "payments": {"status": "settled", "amount": 199.0, "currency": "USD"},
                        "orders": {"status": "payment_failed"},
                        "crash_reports": {"trend": "spiking", "severity": "high"},
                    },
                    "constraints": {"refund_window_days": 30, "sla_hours": 4},
                },
                {
                    "case": {
                        "customer_id": "CUST-9002",
                        "order_id": "ORD-70001",
                        "issue_summary": "Refund requested; internal status conflicts; user mentions chargeback threat.",
                        "customer_message": "If this isn't fixed, I’ll file a chargeback. I was charged and got an error.",
                    },
                    "systems": {
                        "payments": {"status": "authorized", "amount": 49.0, "currency": "USD"},
                        "orders": {"status": "unknown"},
                        "crash_reports": {"trend": "steady", "severity": "medium"},
                    },
                    "constraints": {"refund_window_days": 14, "sla_hours": 2},
                },
            ],
        ),
        expected=[
            # Allow partial correctness through step-by-step sequence grading.
            ExpectedAction(type="check_system"),
            ExpectedAction(type="refund"),
            ExpectedAction(type="file_bug", content_contains="crash"),
            ExpectedAction(type="escalate", content_contains="engineering"),
            ExpectedAction(type="reply", content_contains="refund"),
            ExpectedAction(type="resolve"),
        ],
        metadata={
            "domain": "workflow",
            "expected_policy": "Verify internal status, act on refund, document bug, escalate engineering, respond, then resolve.",
            "hidden": {
                "vip_flag": True,
                "urgency": "high",
                "risk_level": "high",
                "vip_reason": "Enterprise contract renewal pending",
            },
        },
    )

    return TaskCatalog(easy=easy, medium=medium, hard=hard)


def list_tasks() -> List[TaskDefinition]:
    catalog = build_task_catalog()
    return [catalog.easy, catalog.medium, catalog.hard]


def get_task(task_id: str) -> TaskDefinition:
    for t in list_tasks():
        if t.id == task_id:
            return t
    raise KeyError(f"Unknown task_id: {task_id}")


def pick_task_by_difficulty(difficulty: Literal["easy", "medium", "hard"]) -> TaskDefinition:
    catalog = build_task_catalog()
    return {"easy": catalog.easy, "medium": catalog.medium, "hard": catalog.hard}[difficulty]
