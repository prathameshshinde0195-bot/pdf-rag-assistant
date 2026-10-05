import re
import json
import time
import os
from typing import Generator, List, Tuple, Any, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

from utils.ollama_manager import get_installed_model_names


def reciprocal_rank_fusion(
    vector_results: List[Document],
    bm25_results: List[Tuple[Document, float]],
    k: int = 60,
    top_n: int = 4,
) -> List[Tuple[Document, float]]:
    """
    High-speed Reciprocal Rank Fusion (RRF) combining dense vector search and BM25 keyword matching.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    # Rank vector results
    for rank, doc in enumerate(vector_results):
        doc_key = f"{doc.metadata.get('source_name', '')}_{doc.metadata.get('page', '')}_{doc.page_content[:60]}"
        doc_map[doc_key] = doc
        scores[doc_key] = scores.get(doc_key, 0.0) + (1.0 / (k + rank + 1))

    # Rank BM25 results
    for rank, (doc, _) in enumerate(bm25_results):
        doc_key = f"{doc.metadata.get('source_name', '')}_{doc.metadata.get('page', '')}_{doc.page_content[:60]}"
        doc_map[doc_key] = doc
        scores[doc_key] = scores.get(doc_key, 0.0) + (1.0 / (k + rank + 1))

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Normalize score between 0.0 and 1.0 for confidence display
    max_score = (1.0 / (k + 1)) * 2
    final_results = []
    for doc_key, score in sorted_docs[:top_n]:
        normalized_confidence = min(1.0, score / max_score)
        final_results.append((doc_map[doc_key], normalized_confidence))

    return final_results


def generate_instant_smart_followups(query: str, answer: str) -> List[str]:
    """
    Zero-latency heuristic follow-up generator (0 milliseconds overhead).
    Avoids making extra blocking LLM calls.
    """
    # Extract candidate nouns or keywords from query
    words = re.findall(r"\b[A-Za-z]{4,}\b", query)
    topic = words[0].title() if words else "this topic"
    
    # Dynamic contextual prompt templates
    suggestions = [
        f"Can you provide a detailed breakdown of {topic}?",
        f"What are the main risks, limitations, or caveats?",
        f"Extract key statistics, dates, or numbers related to this.",
    ]
    return suggestions


def generate_smart_followups(
    query: str,
    answer: str,
    model_name: str = "llama3.2:3b",
) -> List[str]:
    """
    Fast instant follow-up generator.
    """
    return generate_instant_smart_followups(query, answer)


def retrieve_context_hybrid(
    query: str,
    vector_db,
    bm25_index=None,
    top_k: int = 4,
    rag_mode: str = "Hybrid (Dense + BM25)",
) -> List[Tuple[Document, float]]:
    """
    High-speed retrieval without blocking LLM calls:
    - "Hybrid (Dense + BM25)": Vector + Keyword fusion (RRF)
    - "Dense FAISS Vector": Pure dense embedding similarity
    """
    # 1. Dense retrieval
    vector_docs = []
    try:
        vector_docs = vector_db.similarity_search(query, k=top_k * 2)
    except Exception:
        pass

    # 2. Sparse BM25 retrieval
    bm25_results = []
    if bm25_index is not None and rag_mode in ["Hybrid (Dense + BM25)", "HyDE (Hypothetical Embeddings)"]:
        try:
            bm25_results = bm25_index.search(query, top_k=top_k * 2)
        except Exception:
            pass

    # Combine with RRF if hybrid or bm25 available
    if bm25_results and rag_mode != "Dense FAISS Vector":
        return reciprocal_rank_fusion(vector_docs, bm25_results, top_n=top_k)
    else:
        # Vector only fallback
        return [(doc, 0.85) for doc in vector_docs[:top_k]]


def format_context_for_prompt(retrieved_docs: List[Tuple[Document, float]]) -> str:
    """
    Builds clean, structured context string with document tags, page numbers, and confidence metrics.
    """
    formatted_pieces = []
    for idx, (doc, conf) in enumerate(retrieved_docs):
        source = doc.metadata.get("source_name", "Document")
        page = doc.metadata.get("page", 1)
        conf_pct = int(conf * 100)
        formatted_pieces.append(
            f"--- [Document #{idx+1}: {source} (Page/Section: {page}) | Match Confidence: {conf_pct}%] ---\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_pieces)


def stream_rag_answer(
    query: str,
    chat_history: List[BaseMessage],
    vector_db,
    bm25_index=None,
    model_name: str = "llama3.2:3b",
    temperature: float = 0.1,
    top_k: int = 4,
    rag_mode: str = "Hybrid (Dense + BM25)",
    enable_query_rewrite: bool = False,
    groq_api_key: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    High-Speed Single-Pass Streamlined RAG Pipeline:
    1. Direct Instant Hybrid Retrieval (zero extra LLM calls).
    2. Real-time token streaming with instant Time-to-First-Token (TTFT).
    3. Live DeepSeek-R1 / Reasoning <think> extraction.
    4. Instant zero-latency follow-up questions.
    """
    start_time = time.time()

    # Step 1: Direct Instant Retrieval
    retrieved_items = retrieve_context_hybrid(
        query=query,
        vector_db=vector_db,
        bm25_index=bm25_index,
        top_k=top_k,
        rag_mode=rag_mode,
    )

    formatted_sources = []
    for doc, conf in retrieved_items:
        formatted_sources.append({
            "source_name": doc.metadata.get("source_name", "Document"),
            "page": doc.metadata.get("page", 1),
            "file_type": doc.metadata.get("file_type", "Document"),
            "confidence": round(conf * 100, 1),
            "content": doc.page_content[:450] + ("..." if len(doc.page_content) > 450 else ""),
        })

    # Yield retrieved context metadata immediately (< 0.2s)
    yield {
        "type": "sources",
        "sources": formatted_sources,
        "standalone_query": query,
    }

    if not retrieved_items:
        yield {
            "type": "answer_chunk",
            "chunk": "⚠️ No relevant information was found in the uploaded documents to answer your question.",
            "is_thinking": False,
        }
        return

    context_str = format_context_for_prompt(retrieved_items)

    # Step 2: System prompt optimized for speed & conciseness
    system_prompt = (
        "You are an elite AI Document Analyst.\n"
        "Answer the user question accurately, clearly, and concisely based strictly on the provided context.\n"
        "If the answer cannot be found in the context, state that clearly.\n"
        "Use Markdown with bold headers and bullet points for readability."
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:"),
    ])

    # Initialize LLM (supports local Ollama or Free Cloud Groq)
    num_threads = os.cpu_count() or 4
    if groq_api_key or model_name.startswith("groq:"):
        try:
            from langchain_groq import ChatGroq
            clean_name = model_name.replace("groq:", "").strip() or "llama-3.3-70b-versatile"
            llm = ChatGroq(
                model_name=clean_name,
                groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
                temperature=temperature,
                streaming=True,
            )
        except Exception:
            llm = ChatOllama(model=model_name, temperature=temperature, streaming=True, num_thread=num_threads, num_ctx=3072)
    else:
        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            streaming=True,
            num_thread=num_threads,
            num_ctx=3072,
        )
    chain = qa_prompt | llm

    # Step 3: Stream tokens with live `<think>` reasoning support
    is_inside_think = False
    think_buffer = ""
    answer_buffer = ""
    token_count = 0

    try:
        for chunk in chain.stream({
            "context": context_str,
            "input": query,
            "chat_history": chat_history[-4:] if chat_history else [],
        }):
            token_text = chunk.content if hasattr(chunk, "content") else str(chunk)
            token_count += 1

            if "<think>" in token_text:
                is_inside_think = True
                token_text = token_text.replace("<think>", "")

            if "</think>" in token_text:
                parts = token_text.split("</think>")
                think_buffer += parts[0]
                yield {
                    "type": "think_chunk",
                    "chunk": parts[0],
                    "is_thinking": True,
                }
                is_inside_think = False
                token_text = parts[1] if len(parts) > 1 else ""

            if is_inside_think:
                think_buffer += token_text
                yield {
                    "type": "think_chunk",
                    "chunk": token_text,
                    "is_thinking": True,
                }
            else:
                if token_text:
                    answer_buffer += token_text
                    yield {
                        "type": "answer_chunk",
                        "chunk": token_text,
                        "is_thinking": False,
                    }

    except Exception as ex:
        yield {
            "type": "error",
            "error": str(ex),
        }

    elapsed_time = time.time() - start_time
    tokens_per_sec = round(token_count / elapsed_time, 1) if elapsed_time > 0 else 0

    yield {
        "type": "metrics",
        "tokens": token_count,
        "elapsed_seconds": round(elapsed_time, 2),
        "tokens_per_sec": tokens_per_sec,
        "full_answer": answer_buffer,
        "full_think": think_buffer,
    }


# Backward compatibility helpers
def get_installed_ollama_models() -> List[str]:
    return get_installed_model_names()


def get_conversational_rag_chain(vector_db, model_name: str = "llama3.2:3b", temperature: float = 0.0, top_k: int = 4):
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOllama(model=model_name, temperature=temperature)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Formulate a standalone question from the chat history and latest user query."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question strictly using the provided context:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)


def get_rag_response(vector_db, user_query: str) -> str:
    chain = get_conversational_rag_chain(vector_db)
    result = chain.invoke({"input": user_query, "chat_history": []})
    return result["answer"]