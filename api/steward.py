"""Phase 1.5 — the validation steward.

A Claude agent that READS the deterministic audit (core/audit.py) and does the
judgment a threshold can't: decide whether a t-stat drop is real decay or a
hostile-market artifact, write a plain-English weekly briefing, and propose
bounded config actions for a human to approve.

Hard boundaries (the repo's principle: facts from Python, judgment from Claude):
  - It NEVER recomputes edges — it reasons over the audit's numbers only.
  - Its action space is exactly: disable / re-enable / watch an EXISTING pattern.
    It cannot invent signals, touch the simulator, or move thresholds.
  - Asymmetric autonomy is structural: re-enable only removes a pattern the
    steward itself previously disabled. Turning a base-rejected pattern back on
    still requires a human code change — the steward physically can't.
  - Propose-only: nothing takes effect until a human clicks Apply, which writes
    a runtime override that build_cfg() merges. Reversible from the same UI.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import anthropic

from audit import latest_audit, audit_history, AUDIT_DB

STEWARD_MODEL = "claude-opus-4-8"
OVERRIDES_PATH = os.path.join(ROOT, "data_cache", "pattern_overrides.json")

STEWARD_SYSTEM = """You are the validation steward for Ultrabull, a disciplined \
technical-analysis system for NSE stocks whose entire value is honest, evidence-based \
signals. Every signal it trades was validated with an event study; the system's own \
research proved those edges DECAY over time.

You are handed a deterministic validation audit (calibration of the outlook, and each \
pattern's forward-return edge split into an earlier window vs. the recent 12 months, \
with t-stats). Your job is the judgment a fixed threshold can't do:

1. Decide whether a pattern's recent weakness is genuine DECAY or an artifact — a thin \
recent sample, or a hostile market stretch that hurt a whole class of longs rather than \
that pattern specifically. Use the calibration and the market-regime gate reading as \
context. A pattern that lost edge in the same window the regime gate's value collapsed \
may be suffering the market, not dying.
2. Write a short plain-English briefing: what changed, what's concerning, what to watch.
3. Propose bounded actions — ONLY on existing patterns, one of:
   - "disable": stop trading it (for a pattern that had a real edge and clearly lost it, \
     |recent t| < 2 or turned significantly negative).
   - "reenable": resume a pattern YOU previously disabled that has recovered.
   - "watch": leave it, note it, revisit next audit (the default for ambiguous or thin cases).

Rules you must follow:
- Reason ONLY from the audit's numbers. Never invent statistics or claim an edge the \
audit doesn't show.
- Be conservative. Disabling is a bigger deal than watching — require a clear, \
significant recent signal (not one noisy quarter) and enough recent trades (n >= 30) \
before proposing "disable". When unsure, propose "watch" and say why.
- Asymmetry: proposing "reenable" or anything that ADDS risk demands a higher bar and \
you must flag it for careful human review.
- Every proposal cites the specific numbers behind it (early t, recent t, n).
- Prefer fewer, well-justified proposals over many. If nothing clearly warrants action, \
propose "watch" for the notable items and say the system looks stable.

You propose; a human approves. Nothing you say takes effect on its own."""


SUBMIT_TOOL = {
    "name": "submit_assessment",
    "description": "Submit the weekly briefing and bounded action proposals.",
    "input_schema": {
        "type": "object",
        "properties": {
            "briefing": {"type": "string",
                         "description": "Plain-English markdown: what changed, what's concerning, what to watch. ~150-300 words."},
            "overall": {"type": "string", "enum": ["stable", "watch", "action_needed"],
                        "description": "One-word health call for the whole system."},
            "proposals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "action": {"type": "string", "enum": ["disable", "reenable", "watch"]},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        "needs_human_review": {"type": "boolean",
                                               "description": "True for reenable or anything that adds risk."},
                        "rationale": {"type": "string",
                                      "description": "Cite the specific numbers (early t, recent t, n) and the decay-vs-artifact call."},
                    },
                    "required": ["pattern", "action", "confidence", "rationale"],
                },
            },
        },
        "required": ["briefing", "overall", "proposals"],
    },
}


# ---------- runtime overrides (the one lever; build_cfg merges these) ----------

def load_overrides() -> dict:
    if not os.path.exists(OVERRIDES_PATH):
        return {"disabled": []}
    try:
        with open(OVERRIDES_PATH) as f:
            d = json.load(f)
        d.setdefault("disabled", [])
        return d
    except Exception:
        return {"disabled": []}


def set_override(pattern: str, disabled: bool) -> dict:
    ov = load_overrides()
    cur = set(ov["disabled"])
    if disabled:
        cur.add(pattern)
    else:
        cur.discard(pattern)  # reenable only removes a steward-disabled pattern
    ov["disabled"] = sorted(cur)
    os.makedirs(os.path.dirname(OVERRIDES_PATH), exist_ok=True)
    with open(OVERRIDES_PATH, "w") as f:
        json.dump(ov, f, indent=1)
    return ov


# ---------- storage ----------

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(AUDIT_DB, timeout=10.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS steward ("
        "  generated_at TEXT PRIMARY KEY,"
        "  audit_at TEXT,"
        "  assessment TEXT NOT NULL)"
    )
    return conn


def latest_steward() -> dict | None:
    conn = _db()
    try:
        row = conn.execute("SELECT assessment FROM steward ORDER BY generated_at DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    return json.loads(row[0]) if row else None


def _save(assessment: dict, generated_at: str, audit_at: str | None) -> None:
    conn = _db()
    try:
        with conn:
            conn.execute("INSERT OR REPLACE INTO steward (generated_at, audit_at, assessment) VALUES (?, ?, ?)",
                         (generated_at, audit_at, json.dumps(assessment)))
    finally:
        conn.close()


# ---------- the run ----------

def run_steward() -> dict:
    audit = latest_audit()
    if audit is None:
        return {"ok": False, "error": "no audit to review yet"}

    # compact the audit for the prompt (drop the full reliability arrays; keep headline)
    slim = {
        "generated_at": audit.get("generated_at"),
        "calibration": {hz: {k: v[k] for k in ("n", "ece", "brier", "brier_baseline",
                                               "overall_predicted", "overall_realized") if k in v}
                        for hz, v in (audit.get("calibration", {}).get("horizons") or {}).items()},
        "pattern_decay": audit.get("pattern_decay"),
        "regime_gate": audit.get("regime_gate"),
        "coverage": audit.get("coverage"),
        "history": audit_history(6),
        "currently_disabled_by_steward": load_overrides()["disabled"],
    }

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=STEWARD_MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=STEWARD_SYSTEM,
        tools=[SUBMIT_TOOL],
        tool_choice={"type": "tool", "name": "submit_assessment"},
        messages=[{"role": "user",
                   "content": "Here is this week's validation audit. Review it and submit your "
                              "assessment.\n\n" + json.dumps(slim, default=str)}],
    )
    assessment = None
    for block in resp.content:
        if block.type == "tool_use" and block.name == "submit_assessment":
            assessment = block.input
            break
    if assessment is None:
        return {"ok": False, "error": "steward did not return an assessment"}

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    assessment["generated_at"] = generated_at
    assessment["audit_at"] = audit.get("generated_at")
    assessment["model"] = resp.model
    _save(assessment, generated_at, audit.get("generated_at"))
    return {"ok": True, "assessment": assessment}


if __name__ == "__main__":
    print(run_steward())
