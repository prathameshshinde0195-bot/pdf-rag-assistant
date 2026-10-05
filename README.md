# ⚡ Supercharged AI Document Assistant (Local Ollama RAG)

A state-of-the-art, privacy-first Document Intelligence and RAG Assistant engineered to **surpass standard cloud AI tools** with **Deep Reasoning (`<think>` traces)**, **Hybrid Search (BM25 + FAISS Dense Embeddings)**, **Multi-Format Ingestion**, and **NotebookLM/Perplexity-style Document Superpowers**.

---

## 🌟 Why This Surpasses Standard AI Tools

| Feature | Standard ChatGPT / Basic RAG | ⚡ Supercharged Local RAG Assistant |
| :--- | :--- | :--- |
| **Privacy & Security** | Data sent to cloud servers | **100% Local & Offline** with Ollama |
| **Deep Reasoning** | Fixed reasoning opacity | **Live `<think>` Chain-of-Thought Inspection** (DeepSeek-R1) |
| **Retrieval Engine** | Simple vector similarity | **Hybrid Retrieval (Dense Vector + BM25 Lexical + RRF)** |
| **File Formats** | Only standard PDFs | **PDF, DOCX, CSV, Excel, TXT, MD, JSON, Code** |
| **Document Superpowers** | Basic chat only | **Executive TL;DR, Action Matrix, Mermaid Mind Maps, AI Quizzes, Cross-Doc Comparison** |
| **Follow-up Prompts** | None | **Perplexity-style Contextual Follow-up Chips** |
| **Model Control** | Closed model API | **Pull & Delete SOTA Local Models (DeepSeek-R1, Llama 3.2, Qwen 2.5) directly in UI** |

---

## 🏆 Recommended Ollama Models (2025/2026 SOTA)

1. **🧠 `deepseek-r1:8b`** (or `deepseek-r1:1.5b` for laptops):
   - Best-in-class reasoning, step-by-step logic, mathematical accuracy, and `<think>` chain-of-thought traces.
2. **⚡ `llama3.2:3b`**:
   - Ultra-fast responses with high factual accuracy.
3. **📊 `qwen2.5:7b`**:
   - Best for structured data, tables, spreadsheets, multilingual documents, and code.
4. **🎯 `nomic-embed-text` / `bge-m3`**:
   - High-accuracy embedding models for vector indexing.

---

## 🛠️ Quick Installation & Setup

### 1. Install & Run Ollama
Download Ollama from [ollama.com](https://ollama.com), then start the server:
```bash
ollama serve
```

### 2. Pull the Recommended Models
```bash
# Pull DeepSeek-R1 (Top Reasoning Model)
ollama pull deepseek-r1:8b

# Pull Llama 3.2 (Fast General Model)
ollama pull llama3.2:3b

# Pull Nomic Embed Text (Embedding Model)
ollama pull nomic-embed-text
```

### 3. Install Python Dependencies
```bash
# Activate virtual environment
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
pdf-rag-chatbot-ollama/
├── app.py                             # State-of-the-art Streamlit UI with Glassmorphism
├── requirements.txt                   # Complete dependencies
├── BEST_MODELS_AND_UPGRADE_GUIDE.md   # Model comparison, recommendations & deletion guide
├── utils/
│   ├── ollama_manager.py              # Ollama API client (Health check, model pulling & deletion)
│   ├── ingest.py                      # Multi-format document loader & SimpleBM25 indexer
│   ├── rag_chain.py                   # Hybrid RAG, Reciprocal Rank Fusion & <think> stream parser
│   └── document_analyzer.py           # Summarizer, Action Matrix, Mermaid Mind Maps & AI Quiz
└── temp_files/                        # Uploaded document staging directory
```

---

## 🌐 100% Free Real-World Deployment

To deploy this project to the public internet completely free, choose one of the options below (see [DEPLOYMENT_GUIDE_FREE.md](file:///c:/Users/prath/OneDrive/Desktop/demo/pdf-rag-chatbot-ollama/pdf-rag-chatbot-ollama/DEPLOYMENT_GUIDE_FREE.md) for full instructions):

1. **24/7 Cloud Deployment (Hugging Face Spaces)**: Use the provided `Dockerfile` to deploy to a free 16 GB RAM cloud container with a permanent public HTTPS URL.
2. **Instant 1-Click Public Link (Cloudflare Tunnel)**: Double-click `share_public_url.bat` to generate a live `https://xxxx.trycloudflare.com` URL in 10 seconds.

