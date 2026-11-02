"""A small RAG pipeline over 16 UNC research policies. One function per stage; see README.md."""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

# Quieter model loading, so the demo output stays readable.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "claude-opus-5-5")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # small, free, runs on a laptop CPU
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge wants this on queries
MAX_CHUNK_WORDS = 300  # roughly 400 tokens
TOP_K = 5  # how many chunks we show the LLM


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


@dataclass(eq=False)  # compare chunks by identity, so they can be dict keys
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


# ---------- The index: everything retrieval needs, built once ----------


class Index:
    def __init__(self):
        self.chunks = build_chunks()
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.vectors = self.embed_chunks()

    def embed_chunks(self) -> np.ndarray:
        """Stage 3: one vector per chunk. Done once and cached; only questions are embedded per ask."""
        texts = [c.text for c in self.chunks]
        key = hashlib.sha1((EMBED_MODEL + "".join(texts)).encode()).hexdigest()[:12]
        cache = Path(f".cache/vectors-{key}.npy")
        if cache.exists():
            return np.load(cache)
        print(f"Embedding {len(texts)} chunks (once; cached in .cache/)...")
        vectors = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        cache.parent.mkdir(exist_ok=True)
        np.save(cache, vectors)
        return vectors

    # ---------- Stage 3: vector search ----------

    def dense_search(self, question: str, k: int = 20) -> list[tuple[Chunk, float]]:
        """Find chunks whose meaning is closest to the question (cosine similarity)."""
        q = self.embedder.encode(QUERY_PREFIX + question, normalize_embeddings=True)
        scores = self.vectors @ q  # vectors are normalized, so the dot product is the cosine
        best = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in best]

    def search(self, question: str, mode: str) -> list[tuple[Chunk, float]]:
        """Top chunks for a question, using one retrieval method (compared in demo.py and eval)."""
        if mode == "dense":
            return self.dense_search(question)[:TOP_K]
        raise ValueError(f"unknown mode {mode}")


MODES = ["dense"]
