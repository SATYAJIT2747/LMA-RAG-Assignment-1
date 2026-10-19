# RAG Assignment: Baseline vs. Re-Ranking

Notebook: [`rag_baseline_vs_rerank.ipynb`](rag_baseline_vs_rerank.ipynb)

- **Document:** `DnD_BasicRules_2018.pdf` (D&D Basic Rules v1.0, 180 pages) → 1113 chunks (~800 chars, 100 overlap)
- **Pipeline 1:** `sentence-transformers/all-MiniLM-L6-v2` + FAISS (cosine) → Top 10 → Top 3 → LLM
- **Pipeline 2:** same Top 10 → `cross-encoder/ms-marco-MiniLM-L-6-v2` → re-sorted Top 3 → LLM
- **LLM:** `gemini-flash-lite-latest` (LangChain `ChatGoogleGenerativeAI`, temperature 0), identical prompt for both pipelines

## Run

```bash
pip install -r requirements.txt
# put GOOGLE_API_KEY=... in a .env file (this folder or a parent folder)
jupyter notebook rag_baseline_vs_rerank.ipynb
```

A GPU is used automatically if available (latency numbers in the notebook were measured on CUDA).
