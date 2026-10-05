import re
import os
import json
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


def _get_fast_llm(model_name: str, temperature: float = 0.1, groq_api_key: Optional[str] = None):
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if api_key or model_name.startswith("groq:"):
        try:
            from langchain_groq import ChatGroq
            clean_name = model_name.replace("groq:", "").strip() or "llama-3.3-70b-versatile"
            return ChatGroq(
                model_name=clean_name,
                groq_api_key=api_key,
                temperature=temperature,
            )
        except Exception:
            pass

    from langchain_ollama import ChatOllama
    num_threads = os.cpu_count() or 4
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_thread=num_threads,
        num_ctx=3072,
    )


def _extract_representative_context(chunks: List[Document], max_chars: int = 6000) -> str:
    """
    Selects balanced document chunks across the entire document set for fast processing.
    """
    if not chunks:
        return ""
    
    total_len = sum(len(c.page_content) for c in chunks)
    if total_len <= max_chars:
        return "\n\n---\n\n".join(
            f"[{c.metadata.get('source_name', 'Doc')} - Page {c.metadata.get('page', 1)}]:\n{c.page_content}"
            for c in chunks
        )

    # Sample uniformly from beginning, middle, and end
    step = max(1, len(chunks) // 6)
    sampled = chunks[::step]
    
    text_pieces = []
    current_len = 0
    for c in sampled:
        piece = f"[{c.metadata.get('source_name', 'Doc')} - Page {c.metadata.get('page', 1)}]:\n{c.page_content}"
        if current_len + len(piece) > max_chars:
            break
        text_pieces.append(piece)
        current_len += len(piece)

    return "\n\n---\n\n".join(text_pieces)


def generate_executive_summary(
    chunks: List[Document],
    model_name: str = "llama3.2:3b",
    temperature: float = 0.1,
    groq_api_key: Optional[str] = None,
) -> str:
    """
    Generates a structured Executive Brief & TL;DR with key takeaways and metrics.
    """
    context = _extract_representative_context(chunks)
    if not context:
        return "No document text available to summarize."

    prompt = PromptTemplate.from_template(
        """You are an executive intelligence analyst. Synthesize the document context below into a concise Executive Brief.

Document Context:
{context}

Format your output in clean Markdown:
# 📑 Executive Brief & Key Takeaways

## 🎯 1. TL;DR (30-Second Overview)
[2-3 sentence overview of the core subject]

## 🔑 2. Main Findings & Core Pillars
- **Key Point 1:** Summary
- **Key Point 2:** Summary
- **Key Point 3:** Summary

## 📊 3. Key Numbers & Metrics
- [List critical statistics, dates, or numbers found in text]

## 💡 4. Strategic Takeaways
- [Actionable conclusions based on the text]"""
    )

    llm = _get_fast_llm(model_name, temperature, groq_api_key=groq_api_key)
    chain = prompt | llm
    response = chain.invoke({"context": context})
    return response.content


def generate_action_items_and_risks(
    chunks: List[Document],
    model_name: str = "llama3.2:3b",
    temperature: float = 0.1,
    groq_api_key: Optional[str] = None,
) -> str:
    """
    Extracts action items, milestones, deadlines, and risk factors from the documents.
    """
    context = _extract_representative_context(chunks)
    if not context:
        return "No document text available."

    prompt = PromptTemplate.from_template(
        """You are an operations and risk analyst. Extract actionable duties, deadlines, and risks from the text below.

Document Context:
{context}

Format your response in Markdown:
# 📋 Action Matrix & Risk Assessment

## ✅ Key Action Items & Tasks
| # | Task Description | Owner / Role | Priority / Context |
|---|------------------|--------------|-------------------|
| 1 | ... | ... | ... |

## 📅 Key Deadlines & Timelines
- [Dates and milestones mentioned]

## ⚠️ Risks & Warnings
- **Risk 1:** Description
- **Risk 2:** Description"""
    )

    llm = _get_fast_llm(model_name, temperature, groq_api_key=groq_api_key)
    chain = prompt | llm
    response = chain.invoke({"context": context})
    return response.content


def generate_mind_map_mermaid(
    chunks: List[Document],
    model_name: str = "llama3.2:3b",
    temperature: float = 0.1,
    groq_api_key: Optional[str] = None,
) -> str:
    """
    Generates an interactive Mermaid.js diagram representing document concepts and relationships.
    """
    context = _extract_representative_context(chunks, max_chars=4000)
    if not context:
        return "graph TD\n  A[No Documents] --> B[Upload files to build Mind Map]"

    prompt = PromptTemplate.from_template(
        """Create a visual Mermaid diagram showing the key concepts and relationships in this text.

Document Context:
{context}

Requirements:
1. Return ONLY valid Mermaid diagram code block.
2. Use `graph TD` orientation.
3. Clean concise node names in quotes (e.g. A["Core Topic"] --> B["Subtopic"]).
4. Structure 6-10 interconnected nodes.

```mermaid
graph TD
    A["Main Subject"] --> B["Key Component 1"]
    A --> C["Key Component 2"]
    B --> D["Outcome"]
```"""
    )

    llm = _get_fast_llm(model_name, temperature, groq_api_key=groq_api_key)
    chain = prompt | llm
    content = chain.invoke({"context": context}).content.strip()

    mermaid_match = re.search(r"```(?:mermaid)?\s*([\s\S]*?)```", content, re.IGNORECASE)
    if mermaid_match:
        return mermaid_match.group(1).strip()
    
    if "graph" in content or "flowchart" in content:
        return content
    
    return "graph TD\n  Root[\"Document Analysis\"] --> Topic1[\"Core Findings\"]\n  Root --> Topic2[\"Key Elements\"]"


def generate_quiz_and_flashcards(
    chunks: List[Document],
    model_name: str = "llama3.2:3b",
    num_questions: int = 3,
    groq_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generates interactive multiple-choice quiz questions with answer explanations.
    """
    context = _extract_representative_context(chunks, max_chars=4000)
    if not context:
        return []

    prompt = PromptTemplate.from_template(
        """Generate {num_questions} multiple-choice questions based strictly on this text.

Document Context:
{context}

Respond ONLY with a valid JSON array of objects with this schema:
[
  {{
    "id": 1,
    "question": "Question text?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_index": 0,
    "explanation": "Why this answer is correct."
  }}
]"""
    )

    llm = _get_fast_llm(model_name, temperature=0.1, groq_api_key=groq_api_key)
    chain = prompt | llm
    try:
        raw_output = chain.invoke({"context": context, "num_questions": num_questions}).content.strip()
        raw_output = re.sub(r"^```json\s*", "", raw_output, flags=re.IGNORECASE)
        raw_output = re.sub(r"^```\s*", "", raw_output)
        raw_output = re.sub(r"\s*```$", "", raw_output)
        
        quiz_data = json.loads(raw_output)
        if isinstance(quiz_data, list):
            return quiz_data
    except Exception:
        pass

    return [
        {
            "id": 1,
            "question": "What is the primary focus of the uploaded document(s)?",
            "options": [
                "The core subject indexed in your knowledge base",
                "Unrelated hypothetical details",
                "Generic background data",
                "None of the above"
            ],
            "correct_index": 0,
            "explanation": "The uploaded documents directly cover the indexed domain topics."
        }
    ]


def generate_document_comparison(
    chunks: List[Document],
    model_name: str = "llama3.2:3b",
    groq_api_key: Optional[str] = None,
) -> str:
    """
    Performs side-by-side comparative analysis of multiple uploaded documents.
    """
    doc_names = list(set(c.metadata.get("source_name", "Unknown") for c in chunks))
    if len(doc_names) < 2:
        return "⚠️ Please upload at least 2 different documents to enable Cross-Document Comparison."

    context = _extract_representative_context(chunks, max_chars=6000)

    prompt = PromptTemplate.from_template(
        """Perform a comparative analysis between the uploaded files: {doc_names}.

Document Context:
{context}

Format your output in Markdown:
# ⚖️ Cross-Document Comparative Report

## 📊 Comparison Matrix
| Aspect | {doc1} | {doc2} |
|--------|--------|--------|
| Focus | ... | ... |
| Key Takeaway | ... | ... |

## 🤝 Common Points & Synergies
- [Where the documents agree]

## ⚡ Key Differences
- [Where the documents differ]"""
    )

    doc1 = doc_names[0]
    doc2 = doc_names[1] if len(doc_names) > 1 else "Other Documents"

    llm = _get_fast_llm(model_name, temperature=0.1, groq_api_key=groq_api_key)
    chain = prompt | llm
    response = chain.invoke({
        "doc_names": ", ".join(doc_names),
        "doc1": doc1,
        "doc2": doc2,
        "context": context
    })
    return response.content
