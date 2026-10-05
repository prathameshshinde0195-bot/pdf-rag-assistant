# 🏆 SOTA Ollama Model Guide & AI Upgrade Architecture

This document details the best-in-class local Ollama models for document retrieval, reasoning, and analysis, along with instructions for pruning obsolete models to reclaim disk space.

---

## 🧠 1. The Best Ollama Models for Document RAG (2025/2026 Rankings)

| Rank & Model | Role & Superpower | VRAM / RAM Required | Why it Beats ChatGPT & Standard Tools |
| :--- | :--- | :--- | :--- |
| **🥇 `deepseek-r1:8b`** | **Deep Reasoning & Chain-of-Thought** | ~5.5 GB VRAM / 8 GB RAM | Solves complex logical puzzles, compares conflicting clauses across contracts, math/financial analysis, and outputs inspectable `<think>` reasoning traces. |
| **⚡ `deepseek-r1:1.5b`** | **Ultra-Lightweight Deep Reasoning** | ~1.8 GB RAM (Runs on any CPU) | Brings reasoning traces to low-power laptops and basic hardware without GPU. |
| **🚀 `llama3.2:3b`** | **Ultra-Fast Instant Retrieval** | ~2.5 GB VRAM / 4 GB RAM | Near-instant generation latency with razor-sharp factual grounding and zero hallucinations. |
| **🌟 `llama3.1:8b`** | **Flagship General Intelligence** | ~5.5 GB VRAM / 8 GB RAM | 128,000 token context window, enterprise-grade summarization, and instruction following. |
| **📊 `qwen2.5:7b` / `14b`** | **Tabular Data, Code & Multilingual** | ~5.2 GB VRAM / 8 GB RAM | The gold standard for financial tables, CSV/Excel spreadsheets, coding scripts, and multi-language documents. |

---

## 🎯 2. The Best Embedding Models for Vector Retrieval

| Embedding Model | Dimension / Context | Strengths |
| :--- | :--- | :--- |
| **`nomic-embed-text`** | 768-dim / 8192 context | Standard high-accuracy embedding model for long documents. |
| **`bge-m3`** | 1024-dim / 8192 context | FlagEmbedding SOTA multilingual model supporting dense, sparse, and multi-vector search. |
| **`mxbai-embed-large`** | 1024-dim / 512 context | Top performer on the MTEB (Massive Text Embedding Benchmark). |

---

## ⬇️ 3. Quick-Start Commands to Pull the Best Models

Run these commands in your terminal to install the recommended stack:

```bash
# 1. Pull the Top Reasoning Model (DeepSeek R1)
ollama pull deepseek-r1:8b

# 2. Pull the Ultra-Fast Light Model (Llama 3.2)
ollama pull llama3.2:3b

# 3. Pull the Best Multilingual & Table Model (Qwen 2.5)
ollama pull qwen2.5:7b

# 4. Pull the State-of-the-Art Embedding Models
ollama pull nomic-embed-text
ollama pull bge-m3
```

> 💡 **Tip:** You can also pull any of these models directly inside the Streamlit app using the **🛠️ Ollama Model Manager** drawer in the sidebar!

---

## 🗑️ 4. How to Delete Previous / Obsolete Models (Free Up Disk Space)

Old models (like `llama2`, `mistral:latest`, `orca-mini`, `vicuna`, or duplicate quantizations) can take up 4 GB to 40 GB of disk space.

### Option A: Via Built-in Streamlit UI
1. Open the sidebar in the app.
2. Expand the **🛠️ Ollama Model Manager** drawer.
3. Under **"Delete Previous / Unused Models"**, select the model and click **"Delete Model"**.

### Option B: Via Terminal CLI
```bash
# List all installed models and their sizes
ollama list

# Delete older/unused models
ollama rm llama2
ollama rm mistral:latest
ollama rm llama3:latest
```

---

## 🚀 5. New Architectural Features Surpassing Standard ChatGPT

1. **🧠 Deep Reasoning `<think>` Mode**:
   - Live stream of internal chain-of-thought logic before answering.
2. **🚀 Hybrid BM25 + Dense FAISS Retrieval (Reciprocal Rank Fusion)**:
   - Eliminates standard vector search blindspots by combining lexical keyword precision with semantic embeddings.
3. **📊 NotebookLM & Perplexity-Style Document Superpowers**:
   - **Executive TL;DR**: One-click summary with core findings and key metrics.
   - **Action Matrix & Risks**: Extracts tasks, milestones, owners, and risks.
   - **Visual Mind Map**: Interactive Mermaid.js concept diagram showing entity relationships.
   - **AI Quiz & Flashcards**: Auto-generates comprehension questions with instant answer reveals.
   - **Cross-Document Comparison**: Side-by-side analysis of agreements, versions, or reports.
4. **💡 Smart Follow-up Suggestions (Perplexity-style)**:
   - Clickable follow-up prompt chips after every answer.
5. **📁 Universal Multi-Format Ingestion**:
   - Full support for PDF, Word (.docx), CSV, Excel (.xlsx), Text (.txt, .md), JSON, and Code.
