# Minor Project — RAG Q&A over the Bhagavad Gita

A Retrieval-Augmented Generation (RAG) question-answering app built with **Streamlit**, **LangChain**, **FAISS**, and **Hugging Face** models. The bundled dataset is 641 Bhagavad Gita verses (all 18 chapters) with English translations; a couple of ML/NLP text files are included as generic-document test data.

The project ships two Streamlit apps that trade off speed vs. depth:

| App | File | Port (default) | How it answers |
|---|---|---|---|
| **Fast mode** | `rag_qa_app/fast_app.py` | 8503 | FAISS similarity search returns the top matching verses instantly, then `google/flan-t5-small` generates a short summary of them. |
| **Full RAG** | `rag_qa_app/app.py` | 8502 | LangChain `RetrievalQA` chain: FAISS retriever → `google/flan-t5-small` text2text-generation pipeline, with a keyword-based "quick answer" shortcut and full source-document display. |

Both apps embed documents with `sentence-transformers/all-MiniLM-L6-v2` and persist their FAISS indexes locally (`faiss_index/` and `faiss_index_fast/`) so subsequent runs skip re-indexing.

## Project layout

```
Minor_Project/
└── rag_qa_app/
    ├── app.py                 # Full RAG app (LangChain RetrievalQA + FLAN-T5)
    ├── fast_app.py             # Fast search + AI summary app
    ├── requirements.txt
    ├── data/
    │   ├── bhagavad_gita_verses.csv   # 641 verses, 18 chapters
    │   ├── machine_learning_intro.txt
    │   └── nlp_and_llms.txt
    ├── faiss_index/            # Persisted index for app.py (auto-generated)
    ├── faiss_index_fast/       # Persisted index for fast_app.py (auto-generated)
    ├── START_HERE.bat          # Windows launcher for app.py (port 8502)
    └── FAST_START.bat          # Windows launcher for fast_app.py (port 8503)
```

## How it works

1. **Load data** — `app.py` reads every `.txt` and `.csv` file in `data/`; CSV rows get converted to `Document` objects (with special-cased handling for the Gita's `chapter_title` / `chapter_verse` / `translation` columns). If `data/` has no usable files, it falls back to loading the `om-ashish-soni/vivechan-spritual-text-dataset-v3` dataset from the Hugging Face Hub. `fast_app.py` only reads `bhagavad_gita_verses.csv`.
2. **Chunk & embed** — Documents are split with `RecursiveCharacterTextSplitter` (chunk size 200, overlap 20 in `app.py`) and embedded with `sentence-transformers/all-MiniLM-L6-v2`.
3. **Index** — Embeddings are stored in a FAISS vector store and saved to disk so future runs load the cached index instead of rebuilding it.
4. **Answer**
   - *Fast app*: `vectorstore.similarity_search(question, k=3)` returns the top verses directly, then FLAN-T5 summarizes them.
   - *Full app*: a LangChain `RetrievalQA` chain (`k=1`, `chain_type="stuff"`) retrieves context and FLAN-T5 generates a short answer; source documents are shown in expandable panels. A small keyword dictionary (`krishna`, `dharma`, `karma`, `yoga`, …) can short-circuit common questions for instant replies ("Ultra-Fast Mode").

## Setup

Requires **Python 3.8+**.

```bash
cd Minor_Project/rag_qa_app
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

A venv is optional — the launchers and commands below fall back to the system Python if no `.venv` is present.

## Running

```bash
cd rag_qa_app

# Full RAG app (deeper answers, ~30s per question, more setup work)
python -m streamlit run app.py

# Fast app (instant verse search + quick AI summary)
python -m streamlit run fast_app.py
```

On Windows, double-click the batch launchers instead — they use `.venv` if present, otherwise the system Python:

- `START_HERE.bat` → `app.py` on port 8502
- `FAST_START.bat` → `fast_app.py` on port 8503

Once running, open the printed local URL (e.g. `http://localhost:8502`) and ask questions such as:

- "What does Krishna say about dharma?"
- "Tell me about Chapter 2"
- "What is karma yoga?"

## First run: model download

The first time `app.py` or `fast_app.py` runs, `transformers` downloads `google/flan-t5-small` (~308MB) from the Hugging Face Hub and caches it under `~/.cache/huggingface/hub/`. Subsequent runs load from that cache instantly — no re-download. If the download is slow, `pip install hf_transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1` before launching for a faster transfer.

## Watching what's happening live

Both apps log every stage (index load/create, model load, question received, answer generated, errors) with timestamps to `rag_qa_app/app.log`, in addition to the terminal:

- **In-browser**: `app.py`'s sidebar has a "📊 Live Status" panel showing the last log line and an expandable tail of `app.log` — click "🔄 Refresh" to update it during a demo, no need to switch to the terminal.
- **In a terminal**: tail the file directly while the app runs —
  ```powershell
  Get-Content -Wait -Tail 20 rag_qa_app\app.log
  ```

`app.py` and `fast_app.py` both write to the same `app.log` (in whichever directory they're launched from), so running either — or both — shows up in one combined history.

## Adding your own data

Drop `.txt` or `.csv` files into `rag_qa_app/data/`. `app.py` will pick them up automatically on the next run that doesn't find an existing `faiss_index/` — delete that folder to force reindexing after adding new files.

## Notes

- `faiss_index/` and `faiss_index_fast/` are checked into the repo as prebuilt indexes; delete them if you change the underlying data and want a fresh rebuild.
- The default LLM (`google/flan-t5-small`) is intentionally small for CPU-friendly speed; swap `LLM_MODEL_NAME` in `app.py` for a larger model if you have GPU/accelerate resources and want higher-quality answers.
- `torchvision`/`torchaudio` are not dependencies of this project — if either is installed alongside a mismatched `torch` version, it can break `transformers`/`sentence-transformers` imports (`ModuleNotFoundError: Could not import module 'pipeline'`, `OSError: Could not load this library: ...libtorchaudio.pyd`, or a misleading `Could not import sentence_transformers python package` error). Uninstall the offending package (`pip uninstall torchvision` / `pip uninstall torchaudio`) if you hit one of these.
- `transformers` is pinned to `4.44.2` (not the latest `5.x`) because `app.py`/`fast_app.py` use the `"text2text-generation"` pipeline task with `HuggingFacePipeline`/`RetrievalQA`, which `transformers` v5 restructured/removed (`Unknown task text2text-generation`). If you upgrade `transformers`, expect to also rework the pipeline task name and LangChain integration.
