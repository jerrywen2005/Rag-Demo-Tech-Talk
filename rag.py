"""A small RAG pipeline over 16 UNC research policies. One function per stage; see README.md."""

import json
import os

import anthropic

LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "claude-opus-5-5")


def load_policies():
    return json.load(open("data/policies.json"))


# ---------- The LLM call (used by stages 1 and 6) ----------


def call_llm(system: str, user: str) -> str:
    """Send one message to Claude. Without an API key, print the prompt instead so you can still follow."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return f"(No ANTHROPIC_API_KEY set, so here is the prompt that would be sent.)\n\nSYSTEM:\n{system}\n\nUSER:\n{user}"
    response = anthropic.Anthropic().beta.messages.create(
        model=LLM_MODEL,
        max_tokens=16000,
        output_config={"effort": "low"},  # short Q&A doesn't need deep thinking
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if the model declines, the API retries on a fallback model
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        return "(The model declined to answer.)"
    return "".join(block.text for block in response.content if block.type == "text")


# ---------- Stage 1: the problem. Ask an LLM with no sources ----------


def ask_without_sources(question: str) -> str:
    return call_llm("You help researchers at UNC-Chapel Hill with research policy questions.", question)
