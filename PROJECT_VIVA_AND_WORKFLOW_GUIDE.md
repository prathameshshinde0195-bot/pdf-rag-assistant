# 🎓 Complete Project Guide, Architecture, File Breakdown & Viva Q&A

**Project Title:** Supercharged Privacy-Preserving AI Document Intelligence & Deep Reasoning Assistant using Local RAG (Ollama, DeepSeek-R1, Hybrid BM25 + FAISS, LangChain LCEL, Streamlit)

---

# 📖 TABLE OF CONTENTS
1. [Project Overview & Key Information](#1-project-overview--key-information)
2. [End-to-End Working & Architecture](#2-end-to-end-working--architecture)
3. [Step-by-Step Data Workflow](#3-step-by-step-data-workflow)
4. [File-by-File Detailed Breakdown ("What File Does What")](#4-file-by-file-detailed-breakdown)
5. [Key Technologies & Concepts Explained](#5-key-technologies--concepts-explained)
6. [Top 30 Viva / Interview Questions & Answers](#6-top-30-viva--interview-questions--answers)

---

# 1. Project Overview & Key Information

* **Project Name:** Supercharged AI Document Assistant (Local Ollama RAG)
* **Core Technology:** Hybrid Retrieval-Augmented Generation (Hybrid RAG: Dense Vector + Sparse BM25 + Reciprocal Rank Fusion)
* **LLM Engine:** Local Ollama running **DeepSeek-R1 (8B/1.5B)** for Chain-of-Thought Deep Reasoning and **Meta Llama 3.2 (3B)** / **Llama 3.1 (8B)** / **Qwen 2.5 (7B)** for instant retrieval
* **Embedding Models:** `nomic-embed-text` (8192-token context length) / `bge-m3`
* **Vector Database:** FAISS (Facebook AI Similarity Search - In-Memory Vector Store)
* **Keyword Indexer:** Built-in BM25 Lexical Ranking Engine for exact keyword, numerical, and acronym matching
* **Document Superpowers:** Executive TL;DR Summarizer, Action Items & Risk Matrix, Visual Mermaid.js Mind Maps, AI Quiz Generator, Cross-Document Comparison
* **User Interface:** Streamlit with Glassmorphism, live `<think>` reasoning accordion, source cards with match confidence %, and Perplexity-style follow-up chips
* **Key Innovation:** 100% offline, privacy-first, multi-format (PDF, DOCX, CSV, Excel, TXT, JSON, Code), deep reasoning traces, hybrid retrieval, and built-in model manager with deletion & pulling.

---

# 2. End-to-End Working & Architecture

```text
                                  ┌─────────────────────────────────────────────────────────────┐
                                  │                 STAGE 1: UNIVERSAL INGESTION                │
                                  └─────────────────────────────────────────────────────────────┘
                                                                 │
      [ PDF, DOCX, CSV, XLSX, TXT, JSON, Code ] ───────> [ Multi-Format Universal Parser ]
                                                                 │ (Extracts text, tables + metadata)
                                                                 ▼
                                                  [ RecursiveCharacterTextSplitter ]
                                                  (chunk_size=900, chunk_overlap=150)
                                                                 │
                                ┌────────────────────────────────┴────────────────────────────────┐
                                ▼                                                                 ▼
                  [ OllamaEmbeddings: nomic ]                                      [ SimpleBM25 Sparse Index ]
                    (Dense Semantic Vectors)                                          (Exact Keyword Index)
                                │                                                                 │
                                ▼                                                                 ▼
                     [ FAISS Vector Store ]                                           [ BM25 In-Memory Inverted Index ]

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

                                  ┌─────────────────────────────────────────────────────────────┐
                                  │           STAGE 2: CONVERSATIONAL HYBRID RETRIEVAL          │
                                  └─────────────────────────────────────────────────────────────┘
                                                                 │
   [ User Query ] + [ Conversation History ] ──────────> [ History-Aware Query Rewriter ]
                                                                 │
                                                                 ▼ (Standalone Search Query)
                                ┌────────────────────────────────┴────────────────────────────────┐
                                ▼ (Dense Similarity Search)                       ▼ (Sparse Keyword Search)
                     [ FAISS Vector Top-K ]                                           [ BM25 Score Top-K ]
                                └────────────────────────────────┬────────────────────────────────┘
                                                                 │
                                                                 ▼
                                                [ Reciprocal Rank Fusion (RRF) ]
                                                Score = 1/(60 + rank_dense) + 1/(60 + rank_bm25)
                                                                 │
                                                                 ▼ (Top Ranked Chunks + Confidence %)

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

                                  ┌─────────────────────────────────────────────────────────────┐
                                  │            STAGE 3: REASONING & STREAMED SYNTHESIS          │
                                  └─────────────────────────────────────────────────────────────┘
                                                                 │
                                              [ ChatOllama: DeepSeek-R1 / Llama 3.2 ]
                                                                 │
                                                                 ▼ (Real-Time Token Stream)
                                ┌────────────────────────────────┴────────────────────────────────┐
                                ▼ (<think>...</think> Stream)                                     ▼ (Final Markdown Answer)
                     [ 🧠 Reasoning Accordion ]                                       [ 💬 Answer with Tables & Math ]
                                                                 │
                                                                 ▼
                                                [ Perplexity-Style Smart Follow-Ups ]
```

---

# 3. Step-by-Step Data Workflow

### **Phase 1: Universal Ingestion & Hybrid Indexing**
1. **Upload:** User drops multiple files of any format (PDF, DOCX, CSV, Excel, TXT, MD, JSON).
2. **Signature Verification:** A tuple signature `(filename, size)` prevents unnecessary re-embedding on app reruns.
3. **Structured Extraction:** `utils/ingest.py` parses pages and turns tabular data (CSV/Excel) into LLM-friendly markdown tables.
4. **Hierarchical Splitting:** Splits documents into 900-character chunks with 150-character overlap.
5. **Dual Index Creation:** 
   - **Dense Index:** Vectors computed via `nomic-embed-text` stored in FAISS.
   - **Sparse Index:** Token frequency matrix compiled into in-memory BM25 index.

### **Phase 2: Hybrid Retrieval & Fusion**
1. **Contextual Query Rewriting:** Converts ambiguous queries (e.g., *"What were the risks of that?"*) into unambiguous standalone queries.
2. **Dual-Path Search:** 
   - Dense FAISS retrieves conceptually similar passages.
   - BM25 retrieves exact numerical matches, code identifiers, acronyms, and names.
3. **Reciprocal Rank Fusion (RRF):** Fuses rankings:
   $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
4. **Confidence Normalization:** Converts fused rank into an intuitive 0–100% confidence badge.

### **Phase 3: Deep Reasoning & Streaming Answer**
1. **Prompt Grounding:** Feeds fused context into **DeepSeek-R1** or **Llama 3.2**.
2. **Live Reasoning Extraction:** Detects `<think>...</think>` tokens in real-time, streaming the chain-of-thought into a dedicated collapsible reasoning box.
3. **Markdown Rendering:** Streams final answer with code blocks, tables, and mathematical formulas.
4. **Smart Follow-Ups:** Generates 3 dynamic Perplexity-style follow-up questions.

---

# 4. File-by-File Detailed Breakdown

| File Name | Purpose & Key Responsibilities |
| :--- | :--- |
| **`app.py`** | Main Streamlit interface with Cyberpunk / Glassmorphism dark theme, model selector, file uploader, chat stream, `<think>` accordion, citation cards, and document superpowers toolbar. |
| **`utils/ollama_manager.py`** | Direct Ollama HTTP client: verifies server health, lists models with exact disk sizes in GB, and provides 1-click model pulling and deletion (`DELETE /api/delete`). |
| **`utils/ingest.py`** | Universal document loader (PDF, DOCX, CSV, XLSX, TXT, JSON, Code), `SimpleBM25` ranking class, and recursive text chunking pipeline. |
| **`utils/rag_chain.py`** | Core Hybrid RAG engine: Reciprocal Rank Fusion (RRF), HyDE generation, stream parser for `<think>` reasoning, and Perplexity-style follow-up generator. |
| **`utils/document_analyzer.py`** | NotebookLM-style intelligence tools: Executive TL;DR, Action Matrix & Risk Assessment, Visual Mermaid.js Mind Maps, AI Quiz Generator, and Cross-Document Comparison. |
| **`manage_models.py`** | Standalone Python utility for automated model downloading (DeepSeek-R1, Nomic Embed) and pruning of obsolete models. |
| **`setup_best_models.bat`** | One-click Windows batch script to pull recommended models via CLI. |
| **`requirements.txt`** | Python dependencies including Streamlit, LangChain, FAISS, Pandas, Docx2txt, and OpenPyXL. |

---

# 5. Key Technologies & Concepts Explained

### 1. **Why Hybrid Search (Dense + BM25) Beats Standard Vector Search**
* **Dense Vector Search (Embeddings):** Excels at understanding meaning, synonyms, and themes (e.g. searching "automobile" finds "car"). However, it often fails when looking for exact part numbers, invoice codes, phone numbers, or technical acronyms.
* **BM25 (Sparse Keyword Search):** Matches exact words, terms, and frequencies with mathematical precision.
* **Reciprocal Rank Fusion (RRF):** Merges both results, ensuring the top chunks are both conceptually relevant AND exact.

### 2. **Deep Reasoning (`<think>` Chains of Thought)**
* Models like **DeepSeek-R1** generate internal reasoning traces before answering.
* The system parses these tokens into a dedicated UI accordion, giving the user full visibility into how the AI deduced its conclusions.

### 3. **HyDE (Hypothetical Document Embeddings)**
* For vague user questions, the LLM first dreams up a hypothetical answer paragraph, and then uses that paragraph to search the vector database. Documents that answer the hypothetical scenario are retrieved with high semantic match.

---

# 6. Top 30 Viva / Interview Questions & Answers

### **Q1. What is RAG and why is it superior to fine-tuning for document QA?**
**Answer:** Retrieval-Augmented Generation (RAG) dynamically fetches relevant external document chunks at query time and provides them as context to the LLM. Unlike fine-tuning, RAG prevents hallucinations, provides exact page citations, requires zero retraining when documents change, and keeps private data local.

### **Q2. Why is DeepSeek-R1 considered superior for complex document analysis?**
**Answer:** DeepSeek-R1 utilizes reinforcement learning for deep reasoning and produces step-by-step `<think>` chains of thought. It excels at cross-referencing conflicting clauses, math/data analysis, and multi-step logic where standard LLMs make assumptions.

### **Q3. What is Reciprocal Rank Fusion (RRF) and what formula is used?**
**Answer:** RRF is an algorithm that combines rankings from multiple independent retrieval models (Dense Vector + BM25). Formula:
$$\text{Score}(d) = \sum \frac{1}{k + \text{rank}(d)}$$
where $k=60$ prevents lower-ranked outliers from skewing results.

### **Q4. How does the system handle multi-format files like CSV and Excel?**
**Answer:** `utils/ingest.py` parses tabular datasets into markdown tables and batches them into logical row chunks, allowing the LLM to inspect columns, headers, and individual data rows accurately.

### **Q5. How does the system achieve 100% data privacy?**
**Answer:** All document embeddings and inference run locally through Ollama on `localhost:11434` and in-memory FAISS. No document content, queries, or embeddings are transmitted to external cloud APIs.

### **Q6. What happens when a user asks a follow-up question referencing past messages?**
**Answer:** The **History-Aware Contextualizer** passes the last 6 chat turns and the new question to the LLM to rewrite it into a self-contained, standalone query before executing hybrid retrieval.

### **Q7. How do you delete obsolete Ollama models to free disk space?**
**Answer:** Using the built-in **Ollama Model Manager** in the Streamlit sidebar, or via terminal command `ollama rm <model_name>`, or by running `python manage_models.py`.

### **Q8. What is the role of chunk overlap in text splitting?**
**Answer:** Chunk overlap (e.g. 150 characters) ensures that sentences, names, or ideas that cross chunk boundaries are not cut off, preventing loss of context.

### **Q9. How does the interactive Mermaid Mind Map feature work?**
**Answer:** The LLM inspects representative chunks and structures the core concepts into valid Mermaid.js graph notation (`graph TD`), which Streamlit renders into an interactive knowledge diagram.

### **Q10. What embedding model is used and what is its context length?**
**Answer:** `nomic-embed-text`, featuring a large context window of 8,192 tokens and 768-dimensional dense vectors.
