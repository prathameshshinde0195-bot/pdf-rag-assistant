import sys
import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434"

def check_server():
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False

def list_models():
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return data.get("models", [])
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def pull_model(model_name: str):
    print(f"\n🚀 Downloading & Installing SOTA fast model '{model_name}'...")
    req_data = json.dumps({"name": model_name, "stream": True}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/pull",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=1200) as resp:
            last_pct = -1
            for line in resp:
                if line:
                    data = json.loads(line.decode())
                    status = data.get("status", "")
                    total = data.get("total", 0)
                    completed = data.get("completed", 0)
                    if total > 0:
                        pct = int((completed / total) * 100)
                        if pct != last_pct and pct % 5 == 0:
                            sys.stdout.write(f"\r📥 {status}: {pct}%")
                            sys.stdout.flush()
                            last_pct = pct
                    else:
                        print(f"status: {status}")
            print(f"\n✅ Successfully installed '{model_name}'!\n")
    except Exception as e:
        print(f"\n❌ Error pulling {model_name}: {e}")

def delete_model(model_name: str):
    print(f"🗑️ Deleting old/slow model '{model_name}'...")
    req_data = json.dumps({"name": model_name}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/delete",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="DELETE"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print(f"✅ Deleted '{model_name}' successfully.")
    except Exception as e:
        print(f"❌ Failed to delete {model_name}: {e}")

def main():
    print("=" * 65)
    print("⚡ OLLAMA SOTA MODEL UPGRADE & UNWANTED MODEL CLEANUP UTILITY")
    print("=" * 65)
    
    if not check_server():
        print("❌ Ollama server is NOT running. Please start Ollama ('ollama serve') first!")
        return

    models = list_models()
    print("\n📦 Currently Installed Models:")
    if models:
        for m in models:
            size_gb = round(m.get("size", 0) / (1024 ** 3), 2)
            print(f"  • {m.get('name')} ({size_gb} GB)")
    else:
        print("  (No models currently installed)")

    print("\n" + "=" * 65)
    print("📥 Installing Recommended Fast Stack:")
    print("1. 'llama3.2:3b' (Ultra-Fast 1-2s Response LLM)")
    print("2. 'deepseek-r1:1.5b' (Fast Reasoning LLM with <think> traces)")
    print("3. 'nomic-embed-text' (8k context Document Embedding Model)")
    print("=" * 65)

    pull_model("llama3.2:3b")
    pull_model("deepseek-r1:1.5b")
    pull_model("nomic-embed-text")

    # Clean legacy or heavy slow models
    models_after = list_models()
    keep_list = ["llama3.2:3b", "deepseek-r1:1.5b", "nomic-embed-text", "llama3.2:latest", "deepseek-r1:1.5b-q4_K_M"]
    unwanted = [m["name"] for m in models_after if m["name"] not in keep_list]
    
    if unwanted:
        print("\n" + "=" * 65)
        print("🗑️ Removing Unwanted / Slow Models:", unwanted)
        print("=" * 65)
        for model in unwanted:
            delete_model(model)

    print("\n🎉 Cleanup Complete! Final Active Models:")
    final_models = list_models()
    for m in final_models:
        size_gb = round(m.get("size", 0) / (1024 ** 3), 2)
        print(f"  • {m.get('name')} ({size_gb} GB)")

    print("\n✅ All set! Run 'streamlit run app.py' to use your ultra-fast RAG app.")

if __name__ == "__main__":
    main()
