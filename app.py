import os
import json
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from utils.ollama_manager import (
    check_ollama_status,
    get_installed_models_detailed,
    get_installed_model_names,
    RECOMMENDED_MODELS,
    RECOMMENDED_EMBEDDING_MODELS,
    delete_model,
    pull_model_stream,
)
from utils.ingest import process_files_to_knowledge_base
from utils.rag_chain import (
    stream_rag_answer,
    generate_smart_followups,
)
from utils.document_analyzer import (
    generate_executive_summary,
    generate_action_items_and_risks,
    generate_mind_map_mermaid,
    generate_quiz_and_flashcards,
    generate_document_comparison,
)

load_dotenv()
os.makedirs("temp_files", exist_ok=True)

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="⚡ Supercharged RAG AI • SOTA Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- Custom Styling (Glassmorphism & Cyberpunk Elegance) -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.1rem;
        letter-spacing: -0.5px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-badge.offline {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border-color: rgba(239, 68, 68, 0.3);
    }

    .metric-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 6px 14px;
        border-radius: 10px;
        font-size: 0.85rem;
        color: #e2e8f0;
    }

    .think-box {
        background: rgba(15, 23, 42, 0.7);
        border-left: 3px solid #818cf8;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0 14px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    .source-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        background: rgba(30, 41, 59, 0.75);
    }

    .conf-tag {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Session State Initialization -----------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files_sig" not in st.session_state:
    st.session_state.processed_files_sig = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "loaded_file_names" not in st.session_state:
    st.session_state.loaded_file_names = []
if "active_superpower" not in st.session_state:
    st.session_state.active_superpower = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ----------------- Ollama Health Diagnostic -----------------
ollama_alive, ollama_msg = check_ollama_status()
installed_models_data = get_installed_models_detailed()
installed_model_names = [m["name"] for m in installed_models_data] if installed_models_data else get_installed_model_names()

# ----------------- Top Header Banner -----------------
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div class="gradient-text">⚡ Supercharged AI Document Assistant</div>
            <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
                Ultra-Fast Hybrid RAG (BM25 + FAISS) with Deep Reasoning & Multi-Format Intelligence
            </div>
        </div>
        <div>
            <div class="status-badge {'offline' if not ollama_alive else ''}">
                {'🟢 ' + ollama_msg if ollama_alive else '🔴 ' + ollama_msg}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- Sidebar Configuration -----------------
with st.sidebar:
    st.markdown("### ⚡ Speed & Intelligence Presets")
    
    # Speed Mode Quick Buttons
    speed_preset = st.radio(
        "Select Performance Profile:",
        options=[
            "🚀 Turbo Ultra-Fast (1-2s Answers)",
            "🧠 Deep Reasoning (<think> Mode)",
        ],
        index=0,
        help="Turbo Ultra-Fast uses lightweight models for instant responses. Deep Reasoning uses DeepSeek-R1 for complex multi-step logic.",
    )

    st.markdown("---")
    st.markdown("### 🤖 Active Models")
    st.caption("ℹ️ **2 Models are used in this project:**")
    st.markdown("1. **🎯 Embedding Model:** Embeds document text chunks into vectors.")
    st.markdown("2. **🧠 Chat / Reasoning Model:** Generates the streamed answers.")

    # Auto-adjust model selection based on preset
    default_model_choice = "llama3.2:3b" if "Turbo" in speed_preset else "deepseek-r1:8b"
    if default_model_choice not in installed_model_names:
        if "deepseek-r1:1.5b" in installed_model_names:
            default_model_choice = "deepseek-r1:1.5b"
        elif installed_model_names:
            default_model_choice = installed_model_names[0]

    selected_model = st.selectbox(
        "🧠 Chat LLM Model",
        options=installed_model_names,
        index=installed_model_names.index(default_model_choice) if default_model_choice in installed_model_names else 0,
        help="Select chat model. 'llama3.2:3b' or 'deepseek-r1:1.5b' gives near-instant responses. 'deepseek-r1:8b' gives deep reasoning.",
    )

    # Cloud Inference Option (for Hugging Face / Streamlit Cloud hosting)
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not ollama_alive:
        st.warning("⚠️ Local Ollama is offline. Using Free Cloud Inference fallback.")
        groq_api_key = st.text_input("🔑 Free Groq API Key (from console.groq.com)", value=groq_api_key, type="password", help="100% Free API key for cloud hosting on Streamlit / Hugging Face Spaces.")
        if groq_api_key:
            cloud_models = ["groq:llama-3.3-70b-versatile", "groq:deepseek-r1-distill-llama-70b", "groq:llama-3.1-8b-instant"]
            installed_model_names = cloud_models
            selected_model = st.selectbox("🧠 Cloud LLM Model", options=cloud_models, index=0)

    embedding_model = st.text_input(
        "🎯 Embedding Model",
        value="nomic-embed-text",
        help="Dense embedding model (e.g. nomic-embed-text, bge-m3, all-minilm).",
    )

    # Retrieval Mode selector
    rag_mode = st.radio(
        "🚀 Retrieval Architecture",
        options=[
            "Hybrid (Dense + BM25)",
            "Dense FAISS Vector",
        ],
        index=0,
        help="Hybrid RAG combines dense vector embeddings with BM25 keyword matching for 10x retrieval accuracy with zero extra latency.",
    )

    col_k, col_temp = st.columns(2)
    with col_k:
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=8, value=3, help="Number of chunks retrieved per query.")
    with col_temp:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="0.0 = Strict factual grounding.")

    show_thinking = st.toggle("🧠 Show Thought Process (<think> trace)", value=True if "Reasoning" in speed_preset else False)

    st.markdown("---")
    st.markdown("### 📂 Document Ingestion Hub")
    st.caption("Supports **PDF**, **DOCX**, **TXT**, **MD**, **CSV**, **XLSX**, **JSON**")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "doc", "txt", "md", "csv", "xlsx", "xls", "json", "py"],
        accept_multiple_files=True,
        help="Upload files to build your local high-speed knowledge base.",
    )

    # Ingestion handler with signature caching
    if uploaded_files:
        current_sig = tuple((f.name, f.size) for f in uploaded_files)
        if current_sig != st.session_state.processed_files_sig:
            saved_paths = []
            for file in uploaded_files:
                temp_path = os.path.join("temp_files", file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                saved_paths.append(temp_path)

            with st.spinner(f"🚀 Indexing {len(saved_paths)} document(s)..."):
                try:
                    v_db, bm25_idx, all_chk, chk_count, d_stats = process_files_to_knowledge_base(
                        file_paths=saved_paths,
                        chunk_size=900,
                        chunk_overlap=150,
                        embedding_model=embedding_model,
                    )
                    st.session_state.vector_db = v_db
                    st.session_state.bm25_index = bm25_idx
                    st.session_state.all_chunks = all_chk
                    st.session_state.chunk_count = chk_count
                    st.session_state.doc_stats = d_stats
                    st.session_state.loaded_file_names = [f.name for f in uploaded_files]
                    st.session_state.processed_files_sig = current_sig
                    st.success(f"✅ Indexed {len(saved_paths)} file(s) into {chk_count} neural chunks!")
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")

    if st.session_state.loaded_file_names:
        st.markdown("**Active Knowledge Base:**")
        for fn in st.session_state.loaded_file_names:
            st.markdown(f"- 📄 `{fn}`")
        st.markdown(f"""
        <div class="metric-chip" style="margin-top: 6px; width: 100%; justify-content: center;">
            📊 <strong>{st.session_state.chunk_count}</strong> Chunks Indexed • <strong>{st.session_state.doc_stats.get('total_words', 0):,}</strong> Words
        </div>
        """, unsafe_allow_html=True)

    # ----------------- Ollama Model Manager Section -----------------
    st.markdown("---")
    with st.expander("🛠️ Ollama Model Manager (Pull / Delete)"):
        st.markdown("##### 📥 Pull SOTA Models")
        recommended_pick = st.selectbox(
            "Recommended Models",
            options=list(RECOMMENDED_MODELS.keys()) + list(RECOMMENDED_EMBEDDING_MODELS.keys()),
            help="Select one of the top tested models for document intelligence and reasoning.",
        )
        if recommended_pick in RECOMMENDED_MODELS:
            rec = RECOMMENDED_MODELS[recommended_pick]
            st.caption(f"**{rec['badge']}** • {rec['vram']}")
            st.info(rec["description"])
        
        custom_pull_name = st.text_input("Model to Pull", value=recommended_pick)
        if st.button("⬇️ Pull Model into Ollama", use_container_width=True):
            if custom_pull_name:
                pull_progress_bar = st.progress(0, text=f"Pulling '{custom_pull_name}'...")
                pull_status_text = st.empty()
                try:
                    for status_chunk in pull_model_stream(custom_pull_name):
                        status = status_chunk.get("status", "")
                        total = status_chunk.get("total", 0)
                        completed = status_chunk.get("completed", 0)
                        if total > 0:
                            pct = min(100, int((completed / total) * 100))
                            pull_progress_bar.progress(pct, text=f"{status}: {pct}%")
                        else:
                            pull_status_text.caption(f"Status: {status}")
                    pull_progress_bar.progress(100, text=f"✅ '{custom_pull_name}' successfully downloaded!")
                    st.success(f"Model '{custom_pull_name}' is ready to use!")
                    time.sleep(1)
                    st.rerun()
                except Exception as p_err:
                    st.error(f"Failed to pull model: {str(p_err)}")

        st.markdown("##### 🗑️ Delete Previous / Unused Models")
        if installed_models_data:
            model_to_delete = st.selectbox(
                "Select model to delete (frees disk space)",
                options=[m["name"] for m in installed_models_data],
                key="delete_model_select"
            )
            del_size = next((m["size_str"] for m in installed_models_data if m["name"] == model_to_delete), "Unknown")
            st.caption(f"Disk space to reclaim: **{del_size}**")
            if st.button(f"🗑️ Delete '{model_to_delete}'", type="primary", use_container_width=True):
                success, del_msg = delete_model(model_to_delete)
                if success:
                    st.success(del_msg)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(del_msg)
        else:
            st.caption("No installed models found or Ollama is offline.")

    st.markdown("---")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.active_superpower = None
            st.rerun()
    with col_c2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.vector_db = None
            st.session_state.bm25_index = None
            st.session_state.all_chunks = []
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.processed_files_sig = None
            st.session_state.chunk_count = 0
            st.session_state.doc_stats = {}
            st.session_state.loaded_file_names = []
            st.session_state.active_superpower = None
            st.rerun()


# ----------------- Document Superpowers Toolbar -----------------
st.markdown("#### ⚡ Document Superpowers & Instant Intelligence")
sp_col1, sp_col2, sp_col3, sp_col4, sp_col5 = st.columns(5)

with sp_col1:
    if st.button("📝 Executive TL;DR", use_container_width=True):
        if not st.session_state.all_chunks:
            st.warning("⚠️ Please upload documents first.")
        else:
            st.session_state.active_superpower = "summary"
with sp_col2:
    if st.button("📋 Action Matrix & Risks", use_container_width=True):
        if not st.session_state.all_chunks:
            st.warning("⚠️ Please upload documents first.")
        else:
            st.session_state.active_superpower = "actions"
with sp_col3:
    if st.button("🔍 Visual Mind Map", use_container_width=True):
        if not st.session_state.all_chunks:
            st.warning("⚠️ Please upload documents first.")
        else:
            st.session_state.active_superpower = "mindmap"
with sp_col4:
    if st.button("❓ AI Quiz & Flashcards", use_container_width=True):
        if not st.session_state.all_chunks:
            st.warning("⚠️ Please upload documents first.")
        else:
            st.session_state.active_superpower = "quiz"
with sp_col5:
    if st.button("⚖️ Compare Documents", use_container_width=True):
        if not st.session_state.all_chunks:
            st.warning("⚠️ Please upload documents first.")
        else:
            st.session_state.active_superpower = "compare"


# ----------------- Render Active Superpower Drawer -----------------
if st.session_state.active_superpower:
    with st.container():
        st.markdown("---")
        if st.session_state.active_superpower == "summary":
            with st.spinner("Generating Executive Brief & Takeaways..."):
                summary_md = generate_executive_summary(st.session_state.all_chunks, model_name=selected_model)
                st.markdown(summary_md)
        elif st.session_state.active_superpower == "actions":
            with st.spinner("Extracting Action Items, Deadlines & Risks..."):
                actions_md = generate_action_items_and_risks(st.session_state.all_chunks, model_name=selected_model)
                st.markdown(actions_md)
        elif st.session_state.active_superpower == "mindmap":
            with st.spinner("Synthesizing Visual Knowledge Mind Map..."):
                mermaid_code = generate_mind_map_mermaid(st.session_state.all_chunks, model_name=selected_model)
                st.markdown("### 🔍 Document Concept Knowledge Graph")
                st.caption("Visual representation of key interconnected entities and concepts:")
                st.markdown(f"```mermaid\n{mermaid_code}\n```")
        elif st.session_state.active_superpower == "quiz":
            with st.spinner("Generating AI Comprehension Quiz..."):
                quiz_items = generate_quiz_and_flashcards(st.session_state.all_chunks, model_name=selected_model)
                st.markdown("### ❓ AI Document Comprehension Quiz")
                for q in quiz_items:
                    st.markdown(f"**Q{q.get('id', 1)}: {q.get('question', '')}**")
                    options = q.get("options", [])
                    user_choice = st.radio(f"Select answer for Q{q.get('id', 1)}:", options, key=f"quiz_opt_{q.get('id', 1)}")
                    if st.button(f"Reveal Answer for Q{q.get('id', 1)}", key=f"reveal_{q.get('id', 1)}"):
                        correct_idx = q.get("correct_index", 0)
                        correct_ans = options[correct_idx] if correct_idx < len(options) else ""
                        if user_choice == correct_ans:
                            st.success(f"🎉 Correct! {q.get('explanation', '')}")
                        else:
                            st.info(f"💡 Correct Answer: **{correct_ans}**\n\n_{q.get('explanation', '')}_")
                    st.markdown("---")
        elif st.session_state.active_superpower == "compare":
            with st.spinner("Performing Cross-Document Comparative Intelligence..."):
                compare_md = generate_document_comparison(st.session_state.all_chunks, model_name=selected_model)
                st.markdown(compare_md)

        if st.button("✖️ Close Analysis Panel"):
            st.session_state.active_superpower = None
            st.rerun()
        st.markdown("---")


# ----------------- Chat Interface & Workspace -----------------
st.markdown("#### 💬 Conversational Intelligence")

# If no messages yet, show interactive Hero prompt starters
if not st.session_state.messages:
    if st.session_state.loaded_file_names:
        st.markdown("##### 💡 Suggested Questions to Explore:")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if st.button("📌 Summarize the core findings and key metrics", key="qp1", use_container_width=True):
                st.session_state.pending_query = "Summarize the core findings and key metrics of the uploaded documents."
                st.rerun()
            if st.button("⚖️ What are the primary strengths and potential risks?", key="qp2", use_container_width=True):
                st.session_state.pending_query = "What are the primary strengths and potential risks identified in the text?"
                st.rerun()
        with p_col2:
            if st.button("📊 Extract any tabular data, numbers, or dates mentioned", key="qp3", use_container_width=True):
                st.session_state.pending_query = "Extract any tabular data, key numbers, financial figures, or dates mentioned in the documents."
                st.rerun()
            if st.button("💡 Provide a 3-point executive briefing on this topic", key="qp4", use_container_width=True):
                st.session_state.pending_query = "Provide a 3-point executive briefing on the main topic of these files."
                st.rerun()
    else:
        st.info("👈 **Get Started:** Upload one or more documents (PDF, DOCX, CSV, TXT, etc.) in the sidebar to chat with them!")

# Render message history
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("think") and show_thinking:
            with st.expander("🧠 View Deep Reasoning Thought Process (<think> trace)", expanded=False):
                st.markdown(f'<div class="think-box">{message["think"]}</div>', unsafe_allow_html=True)
        
        st.markdown(message["content"])

        if message.get("metrics"):
            m = message["metrics"]
            st.caption(f"⚡ Generated in **{m.get('elapsed_seconds')}s** • **{m.get('tokens_per_sec')}** tokens/sec • **{m.get('tokens')}** tokens")

        if message.get("sources"):
            with st.expander(f"📌 Referenced Citations ({len(message['sources'])} sources)"):
                for s_idx, src in enumerate(message["sources"]):
                    doc_name = src.get("source_name", "Document")
                    page_num = src.get("page", 1)
                    conf = src.get("confidence", 85)
                    st.markdown(f"""
                    <div class="source-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong>Source #{s_idx + 1}: <code>{doc_name}</code> (Page/Section {page_num})</strong>
                            <span class="conf-tag">{conf}% Relevance</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">{src.get('content', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

        if message.get("followups"):
            st.markdown("💡 **Related Follow-up Questions:**")
            f_cols = st.columns(len(message["followups"]))
            for f_idx, followup_text in enumerate(message["followups"]):
                with f_cols[f_idx]:
                    if st.button(f"👉 {followup_text}", key=f"fup_{msg_idx}_{f_idx}", use_container_width=True):
                        st.session_state.pending_query = followup_text
                        st.rerun()


# ----------------- Query Input & Execution -----------------
user_input = st.chat_input("Ask any question about the uploaded document(s)...")

if st.session_state.pending_query and not user_input:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input:
    if st.session_state.vector_db is None:
        st.warning("⚠️ Please upload at least one document in the sidebar to begin.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            think_placeholder = st.empty()
            answer_placeholder = st.empty()
            metrics_placeholder = st.empty()
            sources_placeholder = st.empty()

            retrieved_sources = []
            accumulated_think = ""
            accumulated_answer = ""
            final_metrics = {}

            # Execute fast single-pass stream
            stream_gen = stream_rag_answer(
                query=user_input,
                chat_history=st.session_state.chat_history,
                vector_db=st.session_state.vector_db,
                bm25_index=st.session_state.bm25_index,
                model_name=selected_model,
                temperature=temperature,
                top_k=top_k,
                rag_mode=rag_mode,
                groq_api_key=groq_api_key,
            )

            for event in stream_gen:
                event_type = event.get("type")

                if event_type == "sources":
                    retrieved_sources = event.get("sources", [])
                elif event_type == "think_chunk":
                    accumulated_think += event.get("chunk", "")
                    if show_thinking:
                        think_placeholder.markdown(
                            f'<div class="think-box"><strong>🧠 Reasoning in progress...</strong><br/>{accumulated_think}</div>',
                            unsafe_allow_html=True,
                        )
                elif event_type == "answer_chunk":
                    accumulated_answer += event.get("chunk", "")
                    answer_placeholder.markdown(accumulated_answer + " ▌")
                elif event_type == "metrics":
                    final_metrics = {
                        "tokens": event.get("tokens", 0),
                        "elapsed_seconds": event.get("elapsed_seconds", 0),
                        "tokens_per_sec": event.get("tokens_per_sec", 0),
                    }
                elif event_type == "error":
                    st.error(f"❌ **Ollama Error:** {event.get('error')}\n\nMake sure Ollama is running (`ollama serve`).")

            if accumulated_think and show_thinking:
                think_placeholder.markdown(
                    f'<details><summary style="cursor: pointer; color: #818cf8; font-weight: 600; margin-bottom: 8px;">🧠 View Deep Reasoning Thought Process</summary><div class="think-box">{accumulated_think}</div></details>',
                    unsafe_allow_html=True,
                )
            else:
                think_placeholder.empty()

            answer_placeholder.markdown(accumulated_answer)

            if final_metrics:
                metrics_placeholder.caption(
                    f"⚡ Generated in **{final_metrics['elapsed_seconds']}s** • **{final_metrics['tokens_per_sec']}** tokens/sec • **{final_metrics['tokens']}** tokens"
                )

            if retrieved_sources:
                with sources_placeholder.expander(f"📌 Referenced Citations ({len(retrieved_sources)} sources)"):
                    for s_idx, src in enumerate(retrieved_sources):
                        doc_name = src.get("source_name", "Document")
                        page_num = src.get("page", 1)
                        conf = src.get("confidence", 85)
                        st.markdown(f"""
                        <div class="source-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <strong>Source #{s_idx + 1}: <code>{doc_name}</code> (Page/Section {page_num})</strong>
                                <span class="conf-tag">{conf}% Relevance</span>
                            </div>
                            <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">{src.get('content', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Instant zero-latency follow-up questions
            followup_questions = generate_smart_followups(user_input, accumulated_answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": accumulated_answer,
            "think": accumulated_think,
            "sources": retrieved_sources,
            "metrics": final_metrics,
            "followups": followup_questions,
        })
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=accumulated_answer))
        st.rerun()
