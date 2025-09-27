# Bhagavad Gita Q&A Application

Your local Retrieval-Augmented Generation (RAG) tool for exploring the Bhagavad Gita with both lightning-fast lookup and AI-generated explanations.

## 🚀 Quick Start

### Option A: Fast Search (recommended)

- Double-click `FAST_START.bat`
- Wait for the browser to open at <http://localhost:8503>
- Get instant verse lookups using keyword search (no AI generation)

### Option B: AI Answers

- Double-click `START_HERE.bat`
- Open <http://localhost:8502> once you see the "Local URL" message
- Receive detailed AI-generated responses (first answer may take ~30 seconds)

### Run Manually (Advanced)

```powershell
cd rag_qa_app
python -m streamlit run app.py
```

## 📁 Data Included

- `data/bhagavad_gita_verses.csv`: 641 verses from all 18 chapters
- English translations ready for semantic search
- Supports additional `.txt` and `.csv` files—drop them in the `data/` folder and they are picked up automatically

## 💬 Example Prompts

- "What does Krishna say about dharma?"
- "Explain karma yoga."
- "Summarize Chapter 2."
- "Where is action and duty discussed?"

## 🧠 Features

- Dual launch modes (fast lookup or AI-generated answers)
- FAISS-backed vector search with persistent index
- Source references alongside each answer
- Simple one-click launch scripts for Windows and PowerShell

## 🔧 Troubleshooting

1. Verify Python is installed: `python --version`
2. Install dependencies: `pip install -r requirements.txt`
3. Run via the provided `.bat` or `.ps1` scripts for automatic setup

Enjoy exploring the wisdom of the Bhagavad Gita! 🕉️
