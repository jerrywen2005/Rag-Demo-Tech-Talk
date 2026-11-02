"""A small RAG pipeline over 16 UNC research policies. One function per stage; see README.md."""

import json
import os
import re
from dataclasses import dataclass

import anthropic

LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "claude-opus-5-5")
MAX_CHUNK_WORDS = 300  # roughly 400 tokens


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


# ---------- Stage 2: chunking ----------


@dataclass
class Chunk:
    doc: dict  # the policy this chunk came from (title, office, source_url, ...)
    section: str  # heading path, e.g. "2. Procedure > 2.1 Initial Determination"
    body: str

    @property
    def text(self) -> str:
        """What we embed, keyword-index and show the LLM: a context header plus the body."""
        return f"{self.doc['title']} > {self.section}\n{self.body}"


def split_long(paragraphs: list[str]) -> list[str]:
    """Pack paragraphs into pieces of at most MAX_CHUNK_WORDS, never cutting a paragraph.
    Each piece starts with the last paragraph of the previous one (a small overlap)."""
    pieces, current = [], []
    for para in paragraphs:
        if current and len(" ".join(current + [para]).split()) > MAX_CHUNK_WORDS:
            pieces.append("\n".join(current))
            current = [current[-1]] if len(current[-1].split()) < 100 else []
        current.append(para)
    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_policy(doc: dict) -> list[Chunk]:
    """Split one policy on its headings. A section that fits is one chunk; a long one is split."""
    chunks, path, paragraphs = [], [], []

    def flush():
        if paragraphs and path and path[-1][1] != "Title":
            section = " > ".join(title for _, title in path)
            chunks.extend(Chunk(doc, section, piece) for piece in split_long(paragraphs))
        paragraphs.clear()

    for line in doc["text"].splitlines():
        heading = re.match(r"(#+) (.*)", line)
        if heading:
            flush()
            level = len(heading.group(1))
            path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, heading.group(2))]
        elif line.strip():
            paragraphs.append(line.strip())
    flush()
    return chunks


def build_chunks() -> list[Chunk]:
    return [chunk for doc in load_policies() for chunk in chunk_policy(doc)]
