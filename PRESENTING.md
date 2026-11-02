# Presenting the demo

How to run the live demo in our 20-minute RAG tech talk (Mon Nov 2), including questions.
[README.md](README.md) explains how each stage works. This page covers what to type and what to say.

## The story in one line

A plain LLM gets UNC policy wrong. We fix it one stage at a time, and the eval numbers show each
stage helping.

## Before the talk

- [ ] Run the setup in the README on the laptop you'll present from, then run every command below
      once. That downloads the models and builds `.cache/`, so nothing downloads live.
- [ ] `export ANTHROPIC_API_KEY=...` in the presenting terminal. Check with
      `python demo.py ask "How often do I need to complete FCOI training?"`: you should get an
      answer, not a printed prompt.
- [ ] Rehearse stage 1 and read what the model actually says. We expect it to give the federal
      "every four years" rule instead of UNC's "annually". If it doesn't, pick another question it
      gets wrong and rehearse with that one.
- [ ] Use a large terminal font, a dark background, and a window about 120 columns wide (the
      `search` output is wide).
- [ ] Save backups in case the Wi-Fi or the API fails:
      ```bash
      mkdir -p backup
      python demo.py nosources "How often do I need to complete FCOI training?" > backup/1-nosources.txt
      python demo.py ask "How often do I need to complete FCOI training?" > backup/6-ask.txt
      ```
      Only stages 1 and 6 need the internet. Everything else runs offline.
- [ ] Put the repo link on the last slide so people can follow along later.

## Run of show (suggested split; change it as a team)

| Time | Who | Stage | What happens |
|---|---|---|---|
| 0:00–2:30 | Andy | Why RAG + stage 1 | Nora in one sentence, then a plain LLM gets the training rule wrong |
| 2:30–6:00 | Anton | Stages 2–3 | Chunking, embeddings, a paraphrase found by meaning |
| 6:00–9:30 | Anton | Stages 4–5 | ITAR: keyword wins, hybrid buries it, reranker fixes it |
| 9:30–13:00 | Jerry | Stages 6–7 | Grounded answer with citations; parking question gets referred |
| 13:00–15:30 | Adithi | Stage 8 + wrap-up | Eval table, the gate trade-off, what Nora adds, repo link |
| 15:30–20:00 | All | Q&A | |

## The commands, in order

Copy and paste these. Each block says what to point at.

**Stage 1: the problem (Andy)**
```bash
python demo.py nosources "How often do I need to complete FCOI training?"
```
Point at: it sounds confident, cites nothing, and (if rehearsal confirmed it) gives the federal
four-year rule. UNC's policy says annually. "It has never read UNC's policy. RAG gives it the policy."

**Stage 2: chunking (Anton)**
```bash
python demo.py chunks
```
Point at: 272 chunks. Each one starts with a header (policy title > section path) and never cuts
mid-paragraph. "The header tells the model, and the user, where the text came from."

**Stage 3: embeddings (Anton)**
```bash
python demo.py search "Can I take my lab notebooks with me when I move to another university?" --mode dense
```
Point at: #1 is Research Data Ownership > Use and Transfer of Research Data, though the policy never
says "notebook" or "move". "Embeddings match meaning, not words."

**Stages 4–5: hybrid and reranking (Anton)**
```bash
python demo.py search "Is ITAR relevant to my lab?"
```
Point at, from top to bottom:
1. `[dense]`: the animal-care policy. Wrong. The embedding doesn't know the acronym.
2. `[keyword]`: Export Controls at #1. BM25 matches the rare exact word "ITAR".
3. `[hybrid]`: RRF merged the lists, and Export Controls is in the top 5, but only at #5.
4. `[rerank]`: the cross-encoder reads question and chunk together and moves it to #1.

Optional, if time allows (the opposite case, where keyword search fails):
```bash
python demo.py search "Am I allowed to take my laptop to China for a conference?" --mode keyword
```

**Stage 6: grounded answer (Jerry)**
```bash
python demo.py ask "How often do I need to complete FCOI training?"
```
Point at: the retrieved chunks and their scores, the `[1]` citations in the answer, the source
links at the bottom, and that it now says annually. Briefly show `GROUNDING_PROMPT` in `rag.py`.

**Stage 7: the gate (Jerry)**
```bash
python demo.py ask "How do I get a parking permit?"
```
Point at: every reranker score is about 0, so outcome is `referred`. The LLM was never called, so
the refusal cost nothing and couldn't be a guess.

**Stage 8: evaluation (Adithi)**
```bash
python demo.py eval
```
Point at: top-1 goes from 16/20 (vector search) to 20/20 (with reranking). Hybrid alone scores
*lower* at #1 (14/20) but finds the right chunk in the top 5; the reranker puts it in order.
Then the gate: 6/6 off-topic questions referred, but 3 real questions also referred. "We'd rather
refer than guess. Nora will add a glossary for acronyms and synonyms to win those back." To show
the trade-off, run `python demo.py eval --gate 0.001`: now off-topic questions get answered.

**Wrap-up (Adithi):** the "What a production app adds" list from the README, and the repo link.

## If something breaks

- **No internet or the API fails:** `cat backup/1-nosources.txt` / `cat backup/6-ask.txt`. Every
  other stage runs offline.
- **A command is slow the first time:** the models weren't cached. Keep talking. It's about a
  minute, once.
- **The LLM answer differs from rehearsal:** that's normal, since LLM output varies. Point at the
  citations and the source links, not the exact wording.

## Likely questions

- **Why not LangChain / LlamaIndex?** They hide the exact steps we're teaching. Our pipeline is
  about 400 lines, and the behavior we care about most (cite sources, refuse to guess) has to be
  explicit and tested.
- **Why not a vector database?** At 272 chunks, NumPy is one line. Nora uses pgvector, so vectors
  live in the same Postgres database as the documents, with no extra service to sync.
- **Cosine vs dot product vs Euclidean?** With normalized vectors they give the same ranking.
  Quality comes from chunking, the embedding model, hybrid search and reranking.
- **Why does the eval only test retrieval?** Retrieval is where most failures start, and testing it
  is free and repeatable. Nora's eval will also check answers (correct outcome, the right sources cited).
- **How did you pick the threshold (0.03)?** From the eval: the highest-scoring off-topic question
  scored 0.024, so 0.03 refers all 6. It has to be re-tuned whenever the reranker changes.
- **Why Claude, and does data leave UNC?** The question and the 5 chunks go to the LLM API.
  Embeddings and reranking run locally. UNC hasn't approved an LLM provider for Nora yet, so the LLM
  sits behind one function that we can swap out.
- **Can it handle follow-up questions?** Not yet. A follow-up would first be rewritten into a
  standalone question before retrieval.
