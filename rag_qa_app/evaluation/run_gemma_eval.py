"""Run Gemma-based RAG evaluation with BLEU and BERTScore metrics.

This script loads the persisted FAISS index (or builds it from the local
Bhagavad Gita dataset), runs inference with the Gemma 2 2B Instruct model
on a curated evaluation set, and reports BLEU and BERTScore metrics.

Usage:
    python -m evaluation.run_gemma_eval --model google/gemma-2-2b-it
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import evaluate
import pandas as pd
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "faiss_index"
EVAL_DIR = ROOT_DIR / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
EVAL_DATASET_PATH = EVAL_DIR / "gemma_eval_dataset.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_NAME = "google/gemma-2-2b-it"
TOP_K_RETRIEVAL = 3
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the Bhagavad Gita "
    "using only the provided context. Quote or paraphrase the context and keep the answer concise."
)


def load_or_build_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if INDEX_DIR.exists():
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    documents: List[Document] = []

    txt_files = list(DATA_DIR.glob("*.txt"))
    for path in txt_files:
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(page_content=text, metadata={"source": path.name})
        )

    csv_files = list(DATA_DIR.glob("*.csv"))
    for path in csv_files:
        df = pd.read_csv(path)
        if {"translation", "chapter_verse"}.issubset(df.columns):
            for _, row in df.iterrows():
                content = (
                    f"Chapter: {row.get('chapter_title', 'Unknown')}\n"
                    f"Verse: {row.get('chapter_verse', 'Unknown')}\n"
                    f"Translation: {row.get('translation', '')}"
                )
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": path.name,
                            "chapter": row.get("chapter_number", "Unknown"),
                            "verse": row.get("chapter_verse", "Unknown"),
                        },
                    )
                )
        else:
            text_columns = [col for col in df.columns if df[col].dtype == "object"]
            for idx, row in df.iterrows():
                content_parts = [
                    f"{col}: {row[col]}"
                    for col in text_columns
                    if pd.notna(row[col]) and str(row[col]).strip()
                ]
                if content_parts:
                    documents.append(
                        Document(
                            page_content="\n".join(content_parts),
                            metadata={
                                "source": f"{path.name}_row_{idx}",
                            },
                        )
                    )

    if not documents:
        raise RuntimeError("No documents found to build the vectorstore.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def load_model_and_tokenizer(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[Warning] CUDA device not detected. Running on CPU may be very slow.")

    dtype: torch.dtype
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if device != "cuda":
        model = model.to(device)

    return model, tokenizer


def load_eval_dataset() -> List[Dict[str, str]]:
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


def build_prompt(question: str, context: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Use only the context below to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_answer(
    question: str,
    vectorstore: FAISS,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = build_prompt(question, context, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            top_p=0.9,
            repetition_penalty=1.05,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    source_spans = [
        {
            "source": doc.metadata.get("source"),
            "chapter": doc.metadata.get("chapter"),
            "verse": doc.metadata.get("verse"),
        }
        for doc in docs
    ]

    return {
        "answer": prediction,
        "context": context,
        "sources": source_spans,
    }


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")

    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    bertscore_res = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
    )

    bertscore_summary = {
        "precision": float(sum(bertscore_res["precision"]) / len(bertscore_res["precision"])),
        "recall": float(sum(bertscore_res["recall"]) / len(bertscore_res["recall"])),
        "f1": float(sum(bertscore_res["f1"]) / len(bertscore_res["f1"])),
    }

    return {
        "bleu": bleu_score,
        "bertscore": bertscore_summary,
    }


def run_evaluation(model_name: str) -> Dict[str, Any]:
    start = time.time()
    vectorstore = load_or_build_vectorstore()
    model, tokenizer = load_model_and_tokenizer(model_name)
    eval_set = load_eval_dataset()

    predictions: List[str] = []
    references: List[str] = []
    samples: List[Dict[str, Any]] = []

    for example in eval_set:
        result = generate_answer(example["question"], vectorstore, model, tokenizer)
        prediction = result["answer"]
        predictions.append(prediction)
        references.append(example["reference_answer"])

        samples.append(
            {
                "id": example["id"],
                "question": example["question"],
                "prediction": prediction,
                "reference_answer": example["reference_answer"],
                "reference_source": example["reference_source"],
                "retrieved_sources": result["sources"],
            }
        )

    metrics = compute_metrics(predictions, references)

    elapsed = time.time() - start
    return {
        "model_name": model_name,
        "metrics": metrics,
        "samples": samples,
        "runtime_seconds": elapsed,
    }


def persist_results(results: Dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sanitized_name = results["model_name"].replace("/", "-")
    out_path = RESULTS_DIR / f"{sanitized_name}.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gemma models on the Bhagavad Gita RAG pipeline.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model identifier to evaluate (default: google/gemma-2-2b-it)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_evaluation(args.model)
    output_path = persist_results(results)
    print(json.dumps(results["metrics"], indent=2))
    print(f"Saved detailed results to {output_path}")


if __name__ == "__main__":
    main()
