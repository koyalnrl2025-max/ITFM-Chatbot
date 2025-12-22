# ingest_knowledge.py — Chroma 1.3.6 compatible (PERSISTENT)

import os
from pathlib import Path
from typing import List
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
from pypdf import PdfReader
import docx

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
PERSIST_DIR = PROJECT_ROOT / "local_chroma_db"
UPLOADS_DIR = PROJECT_ROOT / "knowledge_base"

COLLECTION_NAME = "itfm_docs"
EMBED_MODEL = "all-mpnet-base-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# --------------------------------------


def read_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def read_pdf(p: Path) -> str:
    reader = PdfReader(str(p))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(p: Path) -> str:
    doc = docx.Document(str(p))
    return "\n".join(par.text for par in doc.paragraphs)


def read_csv(p: Path) -> str:
    df = pd.read_csv(p)
    return "\n".join(df.astype(str).apply(lambda r: " | ".join(r.values), axis=1))


def chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def main():
    UPLOADS_DIR.mkdir(exist_ok=True)
    files = list(UPLOADS_DIR.glob("*"))

    if not files:
        print("❌ No files found in knowledge_base/")
        return

    print("Loading embedding model:", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Creating Persistent Chroma DB at:", PERSIST_DIR)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    total_chunks = 0

    for p in tqdm(files, desc="Files"):
        if p.suffix == ".txt":
            text = read_txt(p)
        elif p.suffix == ".pdf":
            text = read_pdf(p)
        elif p.suffix == ".docx":
            text = read_docx(p)
        elif p.suffix == ".csv":
            text = read_csv(p)
        else:
            continue

        chunks = chunk_text(text)
        total_chunks += len(chunks)

        docs, metas, ids, embs = [], [], [], []

        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metas.append({"source": p.name, "chunk_id": i})
            ids.append(f"{p.name}::{i}")
            embs.append(embedder.encode(chunk).tolist())

        collection.add(
            documents=docs,
            metadatas=metas,
            ids=ids,
            embeddings=embs,
        )

    print(f"✅ Ingest complete: {total_chunks} chunks")

    print("📁 DB directory contents:")
    for f in PERSIST_DIR.iterdir():
        print(" -", f.name)


if __name__ == "__main__":
    main()
