# cli_bot.py — Stable RAG CLI with Chroma + Ollama (Python SDK)

import sys
from pathlib import Path
from typing import List, Dict

from duckduckgo_search import DDGS
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
PERSIST_DIR = PROJECT_ROOT / "local_chroma_db"
COLLECTION_NAME = "itfm_docs"

EMBED_MODEL = "all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 12
TOP_FINAL = 4
WEB_RESULTS = 3

OLLAMA_MODEL = "qwen3:0.6b"
# --------------------------------------

def log(*args):
    print(*args, file=sys.stderr)

# -------- Load models --------
log("Loading embedding model:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

log("Loading reranker model:", RERANKER_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)

# -------- Open Chroma --------
log("Opening Chroma DB at:", PERSIST_DIR)
client = chromadb.PersistentClient(path=str(PERSIST_DIR))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# -------- Retrieval --------
def semantic_search(query: str) -> List[Dict]:
    q_emb = embedder.encode(query).tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0] if res.get("ids") else [None] * len(docs)
    return [{"id": i, "text": d, "meta": m} for i, d, m in zip(ids, docs, metas)]

# -------- Reranking --------
def rerank(query: str, candidates: List[Dict]) -> List[Dict]:
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["score"] = float(s)
    return sorted(candidates, key=lambda x: x["score"], reverse=True)

# -------- Web Search --------
def web_search(query: str) -> List[Dict]:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=WEB_RESULTS):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("href", "")
                })
    except Exception as e:
        log(f"Web search error: {e}")
    return results

# -------- Ollama Call --------
def call_ollama(prompt: str) -> str:
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        stream=False
    )
    return response.get("response", "").strip()

# -------- Prompt --------
def build_prompt(question: str, docs: List[Dict], web: List[Dict]) -> str:
    ctx = "\n\n".join(
        f"[DOC] {d['meta'].get('source')} | chunk {d['meta'].get('chunk_id')}\n{d['text']}"
        for d in docs
    ) or "No relevant internal documents found."

    webtxt = "\n\n".join(
        f"[WEB] {w['title']}\n{w['body']}\n{w['url']}"
        for w in web
    ) or "No relevant web results."

    return f"""
You are an experienced IT support engineer. Provide clear, step-by-step troubleshooting instructions.

INTERNAL KNOWLEDGE:
{ctx}

WEB RESULTS:
{webtxt}

USER QUESTION:
{question}

Answer with numbered steps.
"""

# -------- BRIDGE FUNCTION (Required for Streamlit) --------
def answer_question(query: str) -> str:
    """
    Connects the Streamlit app to the RAG logic.
    """
    # 1. Retrieve
    docs = semantic_search(query)
    # 2. Rerank
    top_docs = rerank(query, docs)[:TOP_FINAL]
    # 3. Web Search
    web = web_search(query)
    # 4. Prompt
    prompt = build_prompt(query, top_docs, web)
    # 5. Generate
    answer = call_ollama(prompt)
    
    return answer

# -------- Main CLI (Keeps CLI working) --------
def main():
    print("RAG CLI — IT Support Assistant")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            q = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        print("Working: retrieve → rerank → web → generate …")
        
        # Use the same function as Streamlit to ensure consistency
        answer = answer_question(q)

        print("\nANSWER:\n")
        print(answer)
        print("-" * 80)

if __name__ == "__main__":
    main()