
import streamlit as st
import pandas as pd
import numpy as np
import pickle, faiss, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="BBC News Assistant", page_icon="📰", layout="wide")

# ── Load artefacts (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    with open("best_ml_pipeline.pkl", "rb") as f:
        data = pickle.load(f)
    return data["pipeline"], data["label_encoder"]

@st.cache_resource
def load_rag():
    encoder  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index    = faiss.read_index("faiss_index.index")
    articles = pd.read_csv("articles_for_rag.csv")
    t5_tok   = AutoTokenizer.from_pretrained("google/flan-t5-base")
    t5_mdl   = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    t5_mdl.eval()
    return encoder, index, articles, t5_tok, t5_mdl

def t5_generate(model, tokenizer, prompt, max_new_tokens=120):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📰 BBC News Assistant")
st.markdown("**ITC6110 NLP Group Project** — Text classification + RAG-powered Q&A")

tab1, tab2 = st.tabs(["🏷️ Article Classifier", "💬 Sports Q&A (RAG)"])

# ── Tab 1: Classifier ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Article Category Classifier")
    st.markdown("Paste any news article and the model will classify it into one of 5 BBC categories.")

    clf_pipeline, le = load_classifier()
    user_text = st.text_area("Article text", height=200,
                              placeholder="Paste a news article here...")
    if st.button("Classify", type="primary"):
        if user_text.strip():
            pred_id = clf_pipeline.predict([user_text])[0]
            pred_label = le.classes_[pred_id]
            proba = clf_pipeline.predict_proba([user_text])[0]

            col1, col2 = st.columns(2)
            col1.metric("Predicted Category", pred_label.upper())
            col2.metric("Confidence", f"{proba.max()*100:.1f}%")

            prob_df = pd.DataFrame({
                "Category": le.classes_,
                "Probability": proba
            }).sort_values("Probability", ascending=True)
            st.bar_chart(prob_df.set_index("Category"))
        else:
            st.warning("Please enter some text.")

# ── Tab 2: RAG Q&A ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Sports & News Q&A — Powered by RAG")
    st.markdown("Ask any question about sports or news. The system retrieves relevant BBC articles and generates an answer.")

    encoder, faiss_index, articles, t5_tok, t5_mdl = load_rag()

    query = st.text_input("Your question", placeholder="e.g. Who is Roger Federer?")
    k_val = st.slider("Articles to retrieve", min_value=1, max_value=5, value=3)

    if st.button("Ask", type="primary"):
        if query.strip():
            with st.spinner("Retrieving articles and generating answer..."):
                q_vec = encoder.encode([query], normalize_embeddings=True).astype(np.float32)
                scores, indices = faiss_index.search(q_vec, k_val)

                hits = [{"text": articles["text"].iloc[i],
                         "category": articles["category"].iloc[i],
                         "score": float(s)}
                        for s, i in zip(scores[0], indices[0])]

                context = "\n\n".join(
                    f"Article {i+1} [{h['category']}]: {h['text'][:400]}"
                    for i, h in enumerate(hits)
                )
                prompt = (f"You are a knowledgeable news assistant. "
                          f"Answer using only the articles provided.\n\n"
                          f"{context}\n\nQuestion: {query}\nAnswer:")

                answer = t5_generate(t5_mdl, t5_tok, prompt)

            st.success(f"**Answer:** {answer}")

            with st.expander("📄 Retrieved Articles"):
                for i, h in enumerate(hits):
                    st.markdown(f"**Article {i+1}** — *{h['category']}* (score: {h['score']:.3f})")
                    st.write(h["text"][:500] + "...")
                    st.divider()
        else:
            st.warning("Please enter a question.")
