@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo ⚡ UPGRADING OLLAMA TO SOTA FAST MODELS & REMOVING UNWANTED MODELS
echo =====================================================================
echo.

:: 1. Pull the Best Fast Models
echo [1/3] Pulling Llama 3.2 3B (Ultra-Fast 1-2s Response LLM)...
ollama pull llama3.2:3b

echo.
echo [2/3] Pulling DeepSeek-R1 1.5B (Fast Reasoning LLM with <think> mode)...
ollama pull deepseek-r1:1.5b

echo.
echo [3/3] Pulling Nomic Embed Text (8k Context Embedding Model)...
ollama pull nomic-embed-text

echo.
echo =====================================================================
echo 🗑️ CLEANING UP OLD / SLOW / UNWANTED MODELS (Reclaiming Disk Space)
echo =====================================================================
echo.

:: Deleting older/slow heavy models (suppressing error if not installed)
echo Removing slow 8B reasoning model (deepseek-r1:8b)...
ollama rm deepseek-r1:8b 2>nul

echo Removing legacy llama3 models (llama3, llama3:latest, llama3:8b)...
ollama rm llama3 2>nul
ollama rm llama3:latest 2>nul
ollama rm llama3:8b 2>nul

echo Removing legacy llama2 models...
ollama rm llama2 2>nul
ollama rm llama2:latest 2>nul

echo Removing legacy mistral / phi3 / misc models...
ollama rm mistral 2>nul
ollama rm mistral:latest 2>nul
ollama rm phi3 2>nul
ollama rm orca-mini 2>nul

echo.
echo =====================================================================
echo 📦 FINAL ACTIVE MODELS LIST:
echo =====================================================================
ollama list

echo.
echo =====================================================================
echo ✅ Clean Setup Complete! All unwanted models removed.
echo To run the application: streamlit run app.py
echo =====================================================================
echo.
pause
