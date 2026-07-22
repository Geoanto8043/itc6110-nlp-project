
import streamlit as st
import pandas as pd
import numpy as np
import pickle, faiss, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="BBC News Assistant", page_icon="📰", layout="wide")

# ── load + cache the heavy artefacts once per session (not on every click) ────
@st.cache_resource
def load_classifier():
    with open("best_ml_pipeline.pkl", "rb") as f:
        data = pickle.load(f)
    return data["pipeline"], data["label_encoder"]

@st.cache_resource
def load_rag():
    # retriever encoder + FAISS index + article text + Flan-T5 generator
    encoder  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index    = faiss.read_index("faiss_index.index")
    articles = pd.read_csv("articles_for_rag.csv")
    t5_tok   = AutoTokenizer.from_pretrained("google/flan-t5-base")
    t5_mdl   = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    t5_mdl.eval()
    return encoder, index, articles, t5_tok, t5_mdl

def t5_generate(model, tokenizer, prompt, max_new_tokens=120):
    # run the prompt through Flan-T5 and decode the answer back to text
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def scores_to_proba(scores):
    """Convert decision_function scores to displayable probabilities via softmax."""
    # LinearSVC has no predict_proba, so we softmax the decision scores just for the UI bar
    e = np.exp(scores - scores.max())
    return e / e.sum()

st.title("📰 BBC News Assistant")
st.markdown("**ITC6110 NLP Group Project** — Text classification + RAG-powered Q&A")

tab1, tab2 = st.tabs(["🏷️ Article Classifier", "💬 Sports Q&A (RAG)"])

# ── Tab 1: paste an article → predicted BBC category + confidence ─────────────
with tab1:
    st.subheader("Article Category Classifier")
    st.markdown("Paste any news article and the model will classify it into one of 5 BBC categories.")

    clf_pipeline, le = load_classifier()
    user_text = st.text_area("Article text", height=200,
                              placeholder="Paste a news article here...")
    if st.button("Classify", type="primary"):
        if user_text.strip():
            pred_raw = clf_pipeline.predict([user_text])[0]

            # Decode integer label → category string
            if isinstance(pred_raw, (int, np.integer)):
                pred_label = le.inverse_transform([pred_raw])[0]
            else:
                pred_label = str(pred_raw)

            # Decision scores → softmax probabilities (relative confidence, not calibrated)
            scores = clf_pipeline.decision_function([user_text])[0]
            proba  = scores_to_proba(scores)
            confidence = proba.max()

            col1, col2 = st.columns(2)
            col1.metric("Predicted Category", pred_label.upper())
            col2.metric("Confidence", f"{confidence*100:.1f}%")

            # show all 5 category scores as a bar chart
            prob_df = pd.DataFrame({
                "Category":    le.classes_,
                "Score":       proba,
            }).sort_values("Score", ascending=True)
            st.bar_chart(prob_df.set_index("Category"))
        else:
            st.warning("Please enter some text.")

# ── Tab 2: ask a question → RAG retrieves articles + generates an answer ───────
with tab2:
    st.subheader("Sports & News Q&A — Powered by RAG")
    st.markdown("Ask any question about sports or news. The system retrieves relevant BBC articles and generates an answer.")

    encoder, faiss_index, articles, t5_tok, t5_mdl = load_rag()

    query = st.text_input("Your question", placeholder="e.g. Who is Roger Federer?")
    k_val = st.slider("Articles to retrieve", min_value=1, max_value=5, value=3)

    if st.button("Ask", type="primary"):
        if query.strip():
            with st.spinner("Retrieving articles and generating answer..."):
                # embed the query and pull the k nearest articles from FAISS
                q_vec = encoder.encode([query], normalize_embeddings=True).astype(np.float32)
                scores, indices = faiss_index.search(q_vec, k_val)

                hits = [{"text": articles["text"].iloc[i],
                         "category": articles["category"].iloc[i],
                         "score": float(s)}
                        for s, i in zip(scores[0], indices[0])]

                # stitch the retrieved articles into the prompt context
                context = "\n\n".join(
                    f"Article {i+1} [{h['category']}]: {h['text'][:400]}"
                    for i, h in enumerate(hits)
                )
                prompt = (f"You are a knowledgeable news assistant. "
                          f"Answer using only the articles provided.\n\n"
                          f"{context}\n\nQuestion: {query}\nAnswer:")

                answer = t5_generate(t5_mdl, t5_tok, prompt)

            st.success(f"**Answer:** {answer}")

            # let the user inspect the sources behind the answer
            with st.expander("📄 Retrieved Articles"):
                for i, h in enumerate(hits):
                    st.markdown(f"**Article {i+1}** — *{h['category']}* (score: {h['score']:.3f})")
                    st.write(h["text"][:500] + "...")
                    st.divider()
        else:
            st.warning("Please enter a question.")
