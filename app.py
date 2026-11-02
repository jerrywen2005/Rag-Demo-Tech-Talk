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

tab_ask, tab_search, tab_eval = st.tabs(["Ask", "Compare retrieval", "Evaluation"])

# ---------- Ask: stages 1, 6 and 7 ----------
with tab_ask:
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
        st.markdown(f"**Gate:** best reranker score {best:.3f} vs threshold {rag.GATE_THRESHOLD}: {verdict}.")
        with st.expander("The 5 chunks retrieved (what the LLM sees)"):
            for n, (chunk, score) in enumerate(result["hits"], 1):
                st.markdown(f"**[{n}]** score {score:.3f}: {chunk.doc['title']} > {chunk.section}")
                st.text(chunk.body)

# ---------- Compare retrieval: stages 3, 4 and 5 ----------
with tab_search:
    question = question_box("search")
    st.caption("The top 5 chunks from each retrieval method. Scores are on different scales per method.")
    for row in (rag.MODES[:2], rag.MODES[2:]):
        for col, mode in zip(st.columns(2), row):
            with col:
                st.subheader(MODE_NAMES[mode])
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
        st.dataframe([{"method": MODE_NAMES[mode], "ranked #1": f"{top1}/{n}", "in the top 5": f"{top5}/{n}"}
                      for mode, (top1, top5) in result["retrieval"].items()], hide_index=True)

        st.subheader("The confidence gate")
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
