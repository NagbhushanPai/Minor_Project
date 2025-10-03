# RAG Q&A Application

A simple Question-Answering application using Retrieval-Augmented Generation (RAG) with your documents.

## 🚀 Quick Start

### Option 1: Double-click to run

- **Windows**: Double-click `run.bat`
- **PowerShell**: Right-click `run.ps1` → "Run with PowerShell"

### Option 2: Manual command

```bash
# Navigate to the app folder
cd rag_qa_app

# Run the application
python -m streamlit run app.py
```

## 📁 Adding Your Data

1. Place your files in the `data/` folder:

   - **Text files** (`.txt`): Any plain text content
   - **CSV files** (`.csv`): Structured data with columns

2. The app automatically detects and processes:
   - All `.txt` files as documents
   - CSV files with automatic column detection
   - Your current `bhagavad_gita_verses.csv` is ready to use!

## 💡 Usage

1. **Start the app** using one of the methods above
2. **Open your browser** to `http://localhost:8501`
3. **Ask questions** about your documents:
   - "What is the main message of the Bhagavad Gita?"
   - "Tell me about Chapter 2 verses"
   - "What does Krishna say about dharma?"

## 🛠️ Features

- **Automatic document processing**: Supports TXT and CSV files
- **Smart search**: Uses FAISS vector database for fast retrieval
- **Source references**: Shows which documents were used for answers
- **Persistent index**: Faster startup after first run

## 📋 Requirements

- Python 3.8+
- Required packages (auto-installed):
  - streamlit
  - langchain
  - faiss-cpu
  - transformers
  - sentence-transformers
  - pandas

## 🔧 Troubleshooting

If you get errors:

1. Make sure Python is installed: `python --version`
2. Install Streamlit: `pip install streamlit`
3. Use the batch/PowerShell files for automatic setup

## 📊 Current Data

Your app includes Bhagavad Gita verses with:

- Chapter information
- Verse numbers
- English translations
- Ready for spiritual and philosophical questions!
