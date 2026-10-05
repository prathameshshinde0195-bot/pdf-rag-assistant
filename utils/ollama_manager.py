import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Tuple, Generator, Optional


OLLAMA_BASE_URL = "http://localhost:11434"

# Curated State-of-the-art Model Recommendations for Document RAG (2025/2026)
RECOMMENDED_MODELS = {
    "deepseek-r1:8b": {
        "name": "deepseek-r1:8b",
        "category": "🧠 Deep Reasoning (SOTA)",
        "description": "Best-in-class reasoning & chain-of-thought model. Excels at complex analysis, logic, and synthesis.",
        "vram": "~5.5 GB VRAM / RAM",
        "badge": "🔥 TOP PICK - REASONING",
    },
    "deepseek-r1:1.5b": {
        "name": "deepseek-r1:1.5b",
        "category": "🧠 Deep Reasoning (Lightweight)",
        "description": "Ultra-fast reasoning model optimized for low-end laptops, CPU-only systems, or quick testing.",
        "vram": "~1.8 GB RAM",
        "badge": "⚡ LIGHTWEIGHT REASONING",
    },
    "llama3.2:3b": {
        "name": "llama3.2:3b",
        "category": "⚡ Ultra-Fast General Purpose",
        "description": "Meta's lightweight powerhouse. Extremely fast response times with high factual accuracy.",
        "vram": "~2.5 GB VRAM / RAM",
        "badge": "🚀 FASTEST LLM",
    },
    "llama3.1:8b": {
        "name": "llama3.1:8b",
        "category": "🌟 Flagship General Intelligence",
        "description": "Meta's standard 8B model with 128k context window, exceptional retrieval accuracy, and zero hallucinations.",
        "vram": "~5.5 GB VRAM / RAM",
        "badge": "🌟 GOLD STANDARD",
    },
    "qwen2.5:7b": {
        "name": "qwen2.5:7b",
        "category": "📊 Multilingual, Data & Code",
        "description": "Alibaba SOTA model. Best for structured tables, data extraction, coding, and multilingual documents.",
        "vram": "~5.2 GB VRAM / RAM",
        "badge": "📊 BEST FOR TABLES & DATA",
    },
    "mistral:7b": {
        "name": "mistral:7b",
        "category": "⚡ High Speed Instruction",
        "description": "Reliable, crisp answers with strict instruction-following capabilities.",
        "vram": "~4.8 GB VRAM / RAM",
        "badge": "⚡ SOLID WORKHORSE",
    },
}

RECOMMENDED_EMBEDDING_MODELS = {
    "nomic-embed-text": {
        "name": "nomic-embed-text",
        "description": "8192 token context window, high accuracy dense semantic embeddings.",
        "badge": "🔥 RECOMMENDED EMBEDDING",
    },
    "bge-m3": {
        "name": "bge-m3",
        "description": "FlagEmbedding SOTA multilingual multi-granularity dense+sparse embedding model.",
        "badge": "🌐 MULTILINGUAL EMBEDDING",
    },
    "mxbai-embed-large": {
        "name": "mxbai-embed-large",
        "description": "Top-ranking MTEB retrieval performance for English documents.",
        "badge": "📈 TOP MTEB SCORE",
    },
    "all-minilm": {
        "name": "all-minilm",
        "description": "Ultra-lightweight 384-dimension embedding model for instant indexing.",
        "badge": "⚡ ULTRA LIGHTWEIGHT",
    },
}


def check_ollama_status(base_url: str = OLLAMA_BASE_URL) -> Tuple[bool, str]:
    """
    Checks if the local Ollama server is running and reachable.
    """
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                return True, "Ollama is running smoothly (Port 11434)"
    except urllib.error.URLError:
        return False, "Ollama is not running. Please start Ollama via 'ollama serve' or open the Ollama desktop app."
    except Exception as e:
        return False, f"Ollama connection error: {str(e)}"
    return False, "Ollama is unreachable."


def get_installed_models_detailed(base_url: str = OLLAMA_BASE_URL) -> List[Dict[str, Any]]:
    """
    Queries Ollama API for installed models with size, modified time, and details.
    """
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models = data.get("models", [])
                results = []
                for m in models:
                    size_bytes = m.get("size", 0)
                    size_gb = round(size_bytes / (1024 ** 3), 2)
                    size_mb = round(size_bytes / (1024 ** 2), 1)
                    size_str = f"{size_gb} GB" if size_gb >= 1.0 else f"{size_mb} MB"
                    
                    results.append({
                        "name": m.get("name", "unknown"),
                        "model": m.get("model", m.get("name", "")),
                        "size_bytes": size_bytes,
                        "size_str": size_str,
                        "modified_at": m.get("modified_at", ""),
                        "family": m.get("details", {}).get("family", ""),
                        "parameter_size": m.get("details", {}).get("parameter_size", ""),
                        "quantization_level": m.get("details", {}).get("quantization_level", ""),
                    })
                return results
    except Exception:
        pass
    return []


def get_installed_model_names(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    """
    Returns simple list of installed model names. Falls back to recommended defaults if Ollama is not running.
    """
    detailed = get_installed_models_detailed(base_url)
    if detailed:
        return [m["name"] for m in detailed]
    
    # Fallback list of modern models
    return [
        "deepseek-r1:8b",
        "deepseek-r1:1.5b",
        "llama3.2:3b",
        "llama3.1:8b",
        "llama3",
        "qwen2.5:7b",
        "mistral",
    ]


def delete_model(model_name: str, base_url: str = OLLAMA_BASE_URL) -> Tuple[bool, str]:
    """
    Deletes an Ollama model using the Ollama REST API (DELETE /api/delete).
    """
    try:
        req_data = json.dumps({"name": model_name}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/delete",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                return True, f"Successfully deleted model '{model_name}'."
    except urllib.error.HTTPError as he:
        return False, f"Failed to delete model: HTTP {he.code} - {he.reason}"
    except Exception as e:
        return False, f"Failed to delete model '{model_name}': {str(e)}"
    return False, f"Could not delete model '{model_name}'."


def pull_model_stream(model_name: str, base_url: str = OLLAMA_BASE_URL) -> Generator[Dict[str, Any], None, None]:
    """
    Initiates model download and streams status updates from Ollama API (POST /api/pull).
    """
    req_data = json.dumps({"name": model_name, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/pull",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            for line in response:
                if line:
                    try:
                        chunk_str = line.decode("utf-8").strip()
                        if chunk_str:
                            chunk_data = json.loads(chunk_str)
                            yield chunk_data
                    except Exception:
                        continue
    except Exception as e:
        yield {"status": "error", "error": str(e)}
