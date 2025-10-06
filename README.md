# 🔹 n8n Local Hybrid Reranker (Offline AI Reranker)

> **Created by:** [3.14rock](https://github.com/314rock) — maintained by the community  
> **License:** MIT • **Runs on:** n8n v1.0+ (self-hosted, Docker)

---

## What is this?

**Local Hybrid Reranker** is an **offline AI reranking node** for [n8n](https://n8n.io) that reorders documents by query relevance.  
No cloud, no API keys, no data leaks.

It blends three signals:
- **Semantic** (cosine similarity of embeddings, or external score)  
- **Lexical** (BM25 + exact token overlap)  
- **Recency** (exponential decay by document age)

### Why use it?
A privacy-first alternative to cloud rerankers (Cohere/Jina) — ideal for self-hosted RAG.

---

## Features (v2.1)

- ✅ **Offline & private** — your data never leaves the server  
- ✅ **Global DF** — proper BM25 IDF across the whole corpus  
- ✅ **Configurable weights** — semantic / lexical / exact / recency  
- ✅ **Normalization** — `sigmoid` (default, stable) or `minmax` (adaptive per batch)  
- ✅ **Custom stopwords** — domain-specific control  
- ✅ **Learned model (optional)** — logistic regression with user-provided weights  
- ✅ **Batch mode** — scale to large corpora  
- ✅ **Debug mode** — per-signal breakdown for tuning

---

## Install

```bash
cd ~/.n8n/custom
git clone https://github.com/314rock/n8n-nodes-local-reranker.git
docker restart n8n
