# RAG from scratch: answering UNC research policy questions

A small, readable retrieval-augmented generation (RAG) pipeline over 16 public UNC-Chapel Hill
research policies. It's the demo for our COMP 523 tech talk, and it's the same pipeline our project,
**Nora** (the Research Policy Navigator), is building into its "Ask" feature.

There's no framework and no database. It's about 400 lines of Python in three files you can read
top to bottom, and you can follow along on your laptop. Embeddings and reranking run locally for
free. Only the final answer uses an LLM (Claude), and that step is optional: with no API key, the
demo prints the exact prompt it would send.

```
question
 ├─ vector search (embeddings, cosine)  → top 20 ┐
 └─ keyword search (BM25)               → top 20 ┴─ merge (RRF) → rerank (cross-encoder) → top 5
                                                     → gate: best score too low? → refer to an office
                                                     → LLM answers from the 5 chunks and cites them [n]
```

## Setup (do this before the talk: it downloads about 2.5 GB)

You need Python 3.10 or newer.

```bash
git clone <this repo's URL>
cd Rag-Demo-Tech-Talk
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # includes PyTorch, about 1 GB

python demo.py eval                # first run downloads the two models (about 1.2 GB) and embeds the chunks
```

**Optional, for real answers:** get an API key at <https://platform.claude.com> and run
`export ANTHROPIC_API_KEY=sk-ant-...`. Without one, everything except the final LLM answer still works.

## The files

| File | What it is |
|---|---|
| [`ingest.py`](ingest.py) | Stage 0: downloads the policies and saves them as Markdown (already done for you) |
| [`rag.py`](rag.py) | The pipeline itself, one section per stage. **Read this one.** |
| [`demo.py`](demo.py) | Command-line commands that run each stage and print what happened |
| [`data/policies.json`](data/policies.json) | The 16 policies, with their headings kept |
| [`data/eval.json`](data/eval.json) | 26 test questions with the policy that should answer each |

## Walkthrough

Each stage below is one commit in this repo's history (`git log --oneline`), so you can also
`git checkout` any commit and see the code at that stage.

### Stage 0: ingestion (done once, before anyone asks)

`ingest.py` fetches each policy page from policies.unc.edu and converts it to text, **keeping the
headings** (`## 2. Procedure`, `### 2.1 Initial Determination`). Chunking needs them in the next stage.
The output, `data/policies.json`, is committed, so you don't need to run it. Run `python ingest.py`
only if you want fresh copies.

### Stage 1: the problem

```bash
python demo.py nosources "How often do I need to complete FCOI training?"
```

A plain LLM answers confidently from general knowledge. Federal rules say every four years, but
UNC's policy says **annually** (plus some other triggers). The model has never seen UNC's policy and
can't cite anything. RAG fixes this by retrieving the relevant policy text and making the model
answer from it.

### Stage 2: chunking

```bash
python demo.py chunks
```

We can't paste all 16 policies into every prompt, so we split them into **chunks**: small pieces
we can search for and cite. `chunk_policy()` in `rag.py`:

- splits on the policy's headings, so a section that fits becomes one chunk
- splits a long section only between paragraphs, at about 300 words, and repeats the boundary
  paragraph as a small overlap
- puts a **context header** in front of each chunk: `Policy title > 2. Procedure > 2.1 ...`.
  On its own, "Requests must be made through IRBIS" doesn't say which policy it's from. The
  header supplies that, and it also becomes the citation.

Result: 272 chunks. Cutting every N characters instead would split sentences and mix sections.

### Stage 3: embeddings and vector search

```bash
python demo.py search "Can I take my lab notebooks with me when I move to another university?" --mode dense
```

An **embedding** is a vector (here, 384 numbers) that represents what a text means. Similar
meanings give nearby vectors. We embed every chunk once (cached in `.cache/`), then for each question
we embed only the question and take the chunks with the highest cosine similarity. With normalized
vectors, that's a single NumPy line, `vectors @ question_vector`. That's all a vector database does,
plus an index so it scales.

Embeddings match **meaning**. The top result is the Research Data Ownership policy's "Use and
Transfer of Research Data" section, even though that policy never uses the words "notebook",
"move" or "another university".

### Stage 4: keyword search (BM25) and hybrid retrieval

```bash
python demo.py search "Is ITAR relevant to my lab?"
```

This command shows all four methods side by side. Here vector search fails: it returns the
animal-care policy, because the embedding model doesn't know what "ITAR" means. **BM25** keyword
search scores exact word overlap and gives rare words the most weight, so it finds the Export
Controls policy right away. Policies are full of codes like this (ITAR, FCOI, SOP 1101, 45 CFR 46).

The opposite also happens:

```bash
python demo.py search "Am I allowed to take my laptop to China for a conference?"
```

Here vector search finds Export Controls and keyword search doesn't, because the policy never says
"laptop" or "China". So we run both and merge them with **Reciprocal Rank Fusion** (`rrf()` in
`rag.py`): each list gives a chunk `1 / (60 + rank)`, and we add those up. We merge by rank because
cosine and BM25 scores are on different scales and can't be added directly.

### Stage 5: reranking

Look at the `[hybrid]` and `[rerank]` results from the ITAR search again. Hybrid search found the
right chunk but ranked it only 5th. A **cross-encoder** reranker (`bge-reranker-base`) reads the
question and a chunk *together* and scores how well the chunk answers the question. That's more
accurate than comparing two separate embeddings, but too slow to run on all 272 chunks, so we run it
only on the 20 hybrid candidates and keep the top 5. It moves Export Controls to #1.

### Stage 6: generation with citations

```bash
python demo.py ask "How often do I need to complete FCOI training?"
```

The top 5 chunks go to the LLM as numbered sources, along with a **grounding prompt**
(`GROUNDING_PROMPT` in `rag.py`): answer only from the sources, cite `[n]` after every claim, say
when the sources don't cover something, and never make a compliance decision for the user. We parse
the `[n]` citations back into policy sections and links. Compare this answer with stage 1's.

### Stage 7: the confidence gate

```bash
python demo.py ask "How do I get a parking permit?"
```

Retrieval always returns *something*, even for questions the policies don't cover. If the best
reranker score is below `GATE_THRESHOLD`, we **don't call the LLM**. We say we couldn't find it
and refer the user to an office. A guess about research policy is worse than "I don't know, ask
this office", and skipping the LLM makes refusals free. We also flag any answer that cites no sources.

### Stage 8: evaluation

```bash
python demo.py eval
```

Without measurement, "it seems better" is just a guess. The eval runs all 26 questions through
retrieval (no LLM, so it's free) and checks whether the expected policy comes back:

| Method | Expected policy ranked #1 | In the top 5 |
|---|---|---|
| Vector search (stage 3) | 16/20 | 19/20 |
| Keyword search, BM25 (stage 4) | 9/20 | 17/20 |
| Hybrid, RRF (stage 4) | 14/20 | 19/20 |
| **+ Reranking (stage 5)** | **20/20** | **20/20** |

At the gate threshold of 0.03, it answers 17 of the 20 answerable questions and refers all 6
off-topic ones. The 3 it wrongly refers are casually worded (for example, "take my laptop to China").
That's the trade-off: lower the threshold (`python demo.py eval --gate 0.001`) and some off-topic
questions get "answered" instead. We chose never guessing. The eval is also how we picked the
reranker: the smaller `ms-marco-MiniLM-L-6-v2` got only 17/20 at #1, and its scores couldn't
separate off-topic questions from real ones.

## Try it yourself

- Change `MAX_CHUNK_WORDS` in `rag.py`, then rerun `python demo.py eval`. Do the numbers move?
- Add your own question to `data/eval.json`.
- Swap `RERANK_MODEL` back to `cross-encoder/ms-marco-MiniLM-L-6-v2` and rerun the eval. (Its
  scores are on a different scale, so the gate threshold has to change too.)
- Raise `TOP_K` and look at what `ask` sends the LLM.

## What a production app adds

This demo is the core pipeline. In Nora, the same stages run inside a FastAPI + PostgreSQL app, and a few things change:

- **pgvector** replaces the NumPy array, so chunks live in the database next to their documents.
- **Filters:** only active, authoritative policies are searched, plus an optional topic filter.
- **Query handling:** acronyms and synonyms are expanded from a glossary (IRB, COI, PI ...). This
  could help the casually worded questions the gate currently refers.
- **Referrals** go to the office responsible for the topic, not one general office.
- **Fake AI providers in tests**, so tests never call a real model.
- **Logging** of every question, the sources used, the outcome and latency.
- **Eval in CI:** a change to the prompt, chunking or retrieval has to keep the numbers up.

## Troubleshooting

- **`NOTE: Redirects are currently not supported in MacOs`** is a harmless PyTorch message.
- **The first run is slow:** it downloads the models and embeds the chunks. Later runs reuse
  `~/.cache/huggingface` and `.cache/`.
- **`ask` prints a prompt instead of an answer:** set `ANTHROPIC_API_KEY` (see Setup).
- **Use a different Claude model:** `export RAG_LLM_MODEL=claude-haiku-5-5` (cheaper).
