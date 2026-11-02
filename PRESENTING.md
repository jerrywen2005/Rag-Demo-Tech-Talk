# Presenting the demo

How to run our 20-minute RAG tech talk (Mon Nov 2), including questions.
[README.md](README.md) explains how each stage works. This page covers who does what, what to type,
and what to say.

## The talk

| Time (suggested) | Part | Who | Covers |
|---|---|---|---|
| 0:00–1:30 | Intro | Andy | The team, Nora (a policy Q&A assistant for UNC researchers), what the talk covers |
| 1:30–4:00 | Why RAG | Adithi | LLMs don't know your documents and guess confidently; RAG retrieves the right text and makes the model answer from it |
| 4:00–8:00 | Technical | Anton | The pipeline diagram (README top): chunking, embeddings, BM25, RRF, reranking, the gate |
| 8:00–14:00 | RAG demo | Jerry | Live demo, below (recorded video as backup) |
| 14:00–16:30 | Different ways to do RAG / real world | Andy | See "Talking points for Andy" below |
| 16:30–20:00 | Q&A | All | See "Likely questions" |

**Handoffs:**
- Adithi's "LLMs guess" point is proven by the demo's first command, so she can say "Jerry will show
  this live."
- Anton explains the concepts, so the demo doesn't re-explain them. Jerry just points at where each
  one shows up in the output. Use the same names in the slides: chunk, embedding, BM25, RRF,
  reranker, gate.
- The demo ends on the eval numbers and the gate trade-off, which leads into Andy's "what a real
  app adds".

## Before the talk

- [ ] Run the README setup on the presenting laptop, then run every demo command once and click
      through every UI tab once, so the models and `.cache/` exist and nothing downloads live.
- [ ] Check that `.env` contains `GROQ_API_KEY`. Never open `.env` or run `echo $GROQ_API_KEY`
      on screen or on camera.
- [ ] Record the backup video (below).
- [ ] Terminal: use a large font, a dark background, and a window about 120 columns wide (the
      `search` output is wide). Run `clear` between commands.
- [ ] UI: zoom the browser to about 125–150% (Cmd + plus) so the back row can read it, and close
      other tabs.
- [ ] Put the repo link on the last slide so people can follow along later.

## The demo (Jerry, about 6 minutes)

You can run it in the **terminal** or in the **web UI**. Both use the same code and tell the same
story in the same six steps. Pick one for the live demo and practice it. A good split: use the
UI, because its side-by-side answers and tables read well on a projector, and open the terminal
once at the end to show that it's just a few Python commands people can rerun.

**Starting either one:**
```bash
cd Rag-Demo-Tech-Talk
conda deactivate                  # only if your prompt shows (base)
source .venv/bin/activate         # your prompt should start with (.venv)
streamlit run app.py              # for the UI: opens http://localhost:8501
```
The key is read from `.env`, so you don't need to export anything. Open the UI a few minutes
before your part: the first page load spends about 10 seconds loading the models.

### Option A: web UI

| Step | Tab | Do this | Point at |
|---|---|---|---|
| 1. The problem | Ask | Keep the first example (postdoc extension) and "Also ask the LLM with no sources" checked. Click **Ask**. | Left: confident, says 30 days, may invent a policy or office. Right: **90 days**, cited, with a link to the policy. "Same model. The only difference is the sources." |
| 2. Chunks | Ask | Open "The 5 chunks retrieved". | Each chunk has a title > section header and whole paragraphs. That's what the LLM saw. |
| 3. Retrieval | Compare retrieval | Pick "Is ITAR relevant to my lab?". | Vector search: the animal-care policy (wrong). Keyword: Export Controls #1. Hybrid: Export Controls only at #5. Reranker: moves it to #1. |
| 4. Grounded answer | Ask | (Already shown in step 1.) Point back to the citations and the "Gate: passed" line. | The `[1]` in the answer matches the source link. |
| 5. Refusing to guess | Ask | Pick "How do I get a parking permit?". You can uncheck the comparison box. Click **Ask**. | Yellow referral box. "Gate: … failed, so the LLM was not called." |
| 6. Measuring it | Evaluation | Click **Run the evaluation** (about 30 s; click it before your part to warm the cache). Then drag the slider to 0.001. | 16/20 → 20/20 ranked #1 with the reranker. At 0.03: 6/6 off-topic referred, 3 real questions missed. At 0.001: off-topic questions get answered. "We chose to refer rather than guess." |

### Option B: terminal

Copy and paste these, in order. Each block says what to point at.

**1. The problem, LLM without sources (about 1 min)**
```bash
python demo.py nosources "How far in advance must a UNC postdoc request an extension of the five-year maximum term?"
```
Point at: it sounds confident and official, and in rehearsal it said **30 days**, sometimes naming a
policy or office that doesn't exist. The real policy says **90 days**. "It has never read UNC's
policy, so it guessed."

**2. Chunks (about 30 s)**
```bash
python demo.py chunks
```
Point at: 272 chunks. Each one starts with a header (policy title > section path) and never cuts
mid-paragraph.

**3. Retrieval, the key moment (about 1.5 min)**
```bash
python demo.py search "Is ITAR relevant to my lab?"
```
Point at, from top to bottom:
1. `[dense]`: the animal-care policy. Wrong. The embedding doesn't know the acronym "ITAR".
2. `[keyword]`: Export Controls at #1. BM25 matches the rare exact word.
3. `[hybrid]`: RRF merged both lists. Export Controls is in, but only at #5.
4. `[rerank]`: the reranker reads question and chunk together and moves it to #1.

**4. The grounded answer (about 1 min)**
```bash
python demo.py ask "How far in advance must a UNC postdoc request an extension of the five-year maximum term?"
```
Point at: the retrieved chunks and their high scores, **90 days** with a `[1]` citation, and the link
to the policy at the bottom. "Same question as the first command, now right and checkable."

**5. Refusing to guess (about 30 s)**
```bash
python demo.py ask "How do I get a parking permit?"
```
Point at: every reranker score is about 0, so the outcome is `referred`. The LLM was never called,
so the refusal cost nothing and can't be a guess.

**6. Measuring it (about 1.5 min)**
```bash
python demo.py eval
```
Point at: the expected policy at #1 goes from 16/20 with vector search alone to 20/20 with
reranking. Hybrid alone is *lower* at #1 (14/20) but gets the right chunk into the top 5, and the
reranker puts it in order. Then the gate: all 6 off-topic questions referred, but 3 real,
casually worded questions also referred. "We chose to refer rather than guess." That leads into Andy.

If you have spare time: `python demo.py search "Am I allowed to take my laptop to China for a conference?"`
is the opposite case. Vector search finds Export Controls and keyword search doesn't, because the
policy never says "laptop" or "China".

## Recording the backup video

Record after the models are cached and the key is set, so there's no waiting and no secrets on screen.

1. Open the terminal or the UI (zoomed in), then press **Cmd+Shift+5** (macOS) and choose "Record
   Selected Portion" around that window. QuickTime's "New Screen Recording" also works.
2. Run the six steps above in order, using whichever option you'll present live, and say what you'd
   say live. Pause briefly on each output.
3. Keep it under about 6 minutes. If the first `search` or `ask` is slow, cut the wait in editing.
4. Put the video on the presenting laptop itself, not only in the cloud, and test that it plays.

LLM answers vary from run to run, so the video won't match the live run word for word. That's fine.
If it comes up, say so and point at the citations and links.

## If something breaks live

- **The Groq API fails or the Wi-Fi drops:** switch to the video. Steps 2, 3, 5 and 6 work
  offline (in the UI too), so you can also keep going live and play only the video's steps 1 and 4.
- **The UI won't load or freezes:** switch to the terminal (Option B). It shows the same steps.
- **A command is slow:** the models weren't cached. Keep talking. It's about a minute, once.
- **A rate limit error:** Groq's free tier limits requests per minute. Wait a moment or switch to the video.
- **The answer looks different from rehearsal:** normal. Point at the citations, not the wording.

## Talking points for Andy (different ways to do RAG / real world)

Ideas to choose from. Each one maps to what our demo does.

- **Where vectors live:** we used a NumPy array (272 chunks). Real apps use a vector index:
  pgvector inside Postgres (what Nora uses), or a hosted vector database (Pinecone, Weaviate).
- **Frameworks:** LangChain and LlamaIndex package these steps. They're faster to start with but
  hide what happens. We wrote ours by hand to teach it and to control the refusal behavior.
- **Hosted vs local models:** we ran embeddings and reranking locally for free. Hosted options
  (OpenAI or Voyage embeddings, Cohere Rerank) can score higher but send your text to a third party.
  That matters for UNC data.
- **Fancier retrieval:** query rewriting (fix vague questions before searching), contextual
  retrieval (our context headers are a simple version), and agentic RAG (the model decides when and
  what to search).
- **Real world (Nora):** the same pipeline in FastAPI + Postgres, with filters for active
  policies, referrals to the right office per topic, logging every question, and the eval running
  on every change. See "What a production app adds" in the README.

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
- **Why Groq and gpt-oss?** It's free and fast, so anyone can follow along. The LLM is one
  function (`call_llm`), so swapping providers is a small change. UNC hasn't approved an LLM
  provider for Nora yet.
- **Does data leave your machine?** Only the question and the 5 retrieved chunks, sent to the LLM.
  Embeddings and reranking run locally.
- **Can it handle follow-up questions?** Not yet. A follow-up would first be rewritten into a
  standalone question before retrieval.
