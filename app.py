"""A simple web UI for the demo. It uses the same pipeline as demo.py (everything lives in rag.py).

    streamlit run app.py
"""

import streamlit as st

import rag

EXAMPLES = [
    "How far in advance must a UNC postdoc request an extension of the five-year maximum term?",
    "Is ITAR relevant to my lab?",
    "Who can be a PI on a sponsored project?",
    "Can I take my lab notebooks with me when I move to another university?",
    "How do I get a parking permit?",
]
MODE_NAMES = {"dense": "Vector search", "keyword": "Keyword search (BM25)",
              "hybrid": "Hybrid (RRF)", "rerank": "Hybrid + reranker"}
# What each method's score means. Higher is better for all of them, but the scales differ.
MODE_HELP = {
    "dense": "Finds chunks with similar **meaning**. Score: cosine similarity, −1 to 1 (most land 0.4–0.85).",
    "keyword": "Finds chunks sharing the question's **exact words**, rare words counting most. "
               "Score: BM25, 0 and up with no maximum, so 7 or 16 is normal.",
    "hybrid": "Merges the two lists **by rank** (RRF). Score: Σ 1/(60 + rank), at most about 0.033.",
    "rerank": "A cross-encoder reads question and chunk **together** and reorders the hybrid results. "
              "Score: relevance, 0 to 1. The gate uses this one.",
}

st.set_page_config(page_title="UNC Policy RAG Demo", layout="wide")


@st.cache_resource(show_spinner="Loading the models (only the first time)...")
def get_index():
    return rag.Index()


@st.cache_data(show_spinner="Running the evaluation (about 30 seconds)...")
def get_eval():
    return rag.evaluate(get_index())


def hits_table(hits):
    rows = [{"#": n, "score": round(score, 3), "policy": chunk.doc["title"], "section": chunk.section}
            for n, (chunk, score) in enumerate(hits, 1)]
    st.dataframe(rows, hide_index=True, width="stretch")


def question_box(key):
    example = st.selectbox("Pick an example, or type your own below", EXAMPLES, key=f"{key}_example")
    return st.text_input("Question", value=example, key=f"{key}_{example}")


index = get_index()
st.title("Ask UNC research policy")
st.caption("A retrieval-augmented generation (RAG) demo over 16 UNC-Chapel Hill research policies. "
           f"LLM: {rag.LLM_MODEL} via Groq." + ("" if rag.have_llm_key() else " No GROQ_API_KEY set, "
           "so the app shows the prompt instead of an answer."))

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Chunk:** the 16 policies are split by section into 272 chunks (done once).\n"
        "2. **Retrieve:** vector search + keyword search, merged, top 20.\n"
        "3. **Rerank:** a cross-encoder rescores those 20 and keeps the best 5.\n"
        f"4. **Gate:** if the best score is below {rag.GATE_THRESHOLD}, refer to an office instead of guessing.\n"
        "5. **Answer:** the LLM answers only from the 5 chunks and cites them [n]."
    )
    st.caption("Every score: higher = better match. Each method uses its own scale.")

tab_ask, tab_search, tab_eval = st.tabs(["Ask", "Compare retrieval", "Evaluation"])

# ---------- Ask: stages 1, 6 and 7 ----------
with tab_ask:
    st.caption("Ask a question. **Left:** the LLM with no sources (it answers from memory). "
               "**Right:** the RAG pipeline, which answers only from retrieved policy text and cites it.")
    question = question_box("ask")
    compare = st.checkbox("Also ask the LLM with no sources, for comparison", value=True)
    if st.button("Ask", type="primary"):
        left, right = st.columns(2) if compare else (None, st.container())
        if compare:
            with left:
                st.subheader("Without sources")
                with st.spinner("Asking the LLM..."):
                    st.markdown(rag.ask_without_sources(question))
        with right:
            st.subheader("With RAG")
            with st.spinner("Retrieving and answering..."):
                result = rag.ask(index, question)
            if result["outcome"] == "referred":
                st.warning(result["answer"])
            elif result["outcome"] == "answered":
                st.success("Answered from the policies")
                st.markdown(result["answer"])
            else:
                st.info(f"Outcome: {result['outcome']}")
                st.markdown(result["answer"])
            if result["cited"]:
                st.markdown("**Sources**")
                for chunk in result["cited"]:
                    st.markdown(f"- [{chunk.doc['title']}]({chunk.doc['source_url']}) > {chunk.section}")

        best = result["hits"][0][1]
        verdict = "passed, so the LLM was called" if best >= rag.GATE_THRESHOLD else "failed, so the LLM was not called"
        st.markdown(f"**Confidence gate:** best reranker score {best:.3f} vs threshold "
                    f"{rag.GATE_THRESHOLD}: {verdict}.")
        st.caption("The reranker scores how well each chunk answers the question, from 0 to 1 (higher is "
                   "better). If even the best chunk scores below the threshold, the policies probably don't "
                   "cover the question, so we refer to an office instead of letting the LLM guess.")
        with st.expander("The 5 chunks retrieved (what the LLM sees)"):
            st.caption("Numbered as the LLM sees them, so [1] in the answer means chunk [1] here.")
            for n, (chunk, score) in enumerate(result["hits"], 1):
                st.markdown(f"**[{n}]** score {score:.3f}: {chunk.doc['title']} > {chunk.section}")
                st.text(chunk.body)

# ---------- Compare retrieval: stages 3, 4 and 5 ----------
with tab_search:
    st.caption("The same question through each retrieval method, top 5 each. Look at **which policy** "
               "comes out on top. Higher score = better match in every table, but each method uses its own "
               "scale, so compare scores within a table, not across tables.")
    question = question_box("search")
    for row in (rag.MODES[:2], rag.MODES[2:]):
        for col, mode in zip(st.columns(2), row):
            with col:
                st.subheader(MODE_NAMES[mode])
                st.caption(MODE_HELP[mode])
                hits_table(index.search(question, mode))

# ---------- Evaluation: stage 8 ----------
with tab_eval:
    st.caption("26 fixed questions (data/eval.json): 20 the policies answer, 6 they don't. "
               "Retrieval only, no LLM calls.")
    if st.button("Run the evaluation", type="primary") or "eval_ran" in st.session_state:
        st.session_state.eval_ran = True
        result = get_eval()
        n = result["answerable"]
        st.subheader("Is the expected policy found?")
        st.caption(f"For the {n} answerable questions: how often each method ranks the right policy first, "
                   "and how often it's anywhere in the 5 chunks the LLM would see. Higher is better.")
        st.dataframe([{"method": MODE_NAMES[mode], "ranked #1": f"{top1}/{n}", "in the top 5": f"{top5}/{n}"}
                      for mode, (top1, top5) in result["retrieval"].items()], hide_index=True)

        st.subheader("The confidence gate")
        st.caption("If a question's best reranker score is below the threshold, we refer instead of answering. "
                   "**Raise it** and the gate gets stricter: off-topic questions are refused, but so are some "
                   "real ones. **Lower it** and more get answered, including off-topic ones the LLM would guess "
                   f"at. We use {rag.GATE_THRESHOLD}: the highest-scoring off-topic question scored 0.024.")
        threshold = st.slider("Gate threshold (best reranker score below it = refer)",
                              0.0, 0.1, rag.GATE_THRESHOLD, step=0.001, format="%.3f")
        gate = rag.check_gate(result["scores"], threshold)
        a, b = st.columns(2)
        a.metric("Answerable questions answered", f"{gate['answered']}/{n}")
        b.metric("Off-topic questions referred", f"{gate['referred']}/{result['unanswerable']}")
        if gate["mistakes"]:
            st.markdown("**Mistakes at this threshold**")
            st.dataframe([{"question": q, "best score": round(s, 4), "wrongly": what}
                          for q, s, what in gate["mistakes"]], hide_index=True, width="stretch")
