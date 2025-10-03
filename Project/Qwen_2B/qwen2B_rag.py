# qwen2B_rag.py

import os
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# ========================
# NLTK SETUP
# ========================
nltk.download('punkt')

# ========================
# DEVICE CONFIG
# ========================
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ========================
# LOAD MODELS
# ========================
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
rag_model_name = "Qwen/Qwen-2B"

print("Loading embedding model...")
embedder = SentenceTransformer(embed_model_name, device=device)

print("Loading Qwen-2B model...")
tokenizer = AutoTokenizer.from_pretrained(rag_model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    rag_model_name,
    device_map="auto",
    offload_folder="offload",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# ========================
# LOAD DOCUMENTS
# ========================
csv_path = "../rag_qa_app/data/bhagavad_gita_verses.csv"
df = pd.read_csv(csv_path)
if 'translation' not in df.columns:
    raise ValueError("CSV must have a column named 'translation'")

documents = df['translation'].tolist()
print(f"Loaded {len(documents)} verses from CSV.")

# ========================
# FAISS INDEX
# ========================
print("Encoding documents for FAISS...")
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(doc_embeddings)
print(f"Indexed {index.ntotal} documents.")

# ========================
# RETRIEVAL FUNCTION
# ========================
def retrieve(query, k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    return [documents[i] for i in indices[0]]

# ========================
# RAG FUNCTION
# ========================
def rag_answer_local(question, k=3, max_new_tokens=150):
    retrieved_docs = retrieve(question, k)
    context = "\n".join(retrieved_docs)
    prompt = f"Answer the question using ONLY the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer, retrieved_docs

# ========================
# TEST RUN
# ========================
if __name__ == "__main__":
    questions = [
        "What advice did Krishna give Arjuna?",
        "What is the main teaching of Chapter 2?",
        "How should one act without attachment?",
        "What is the purpose of performing one's duties?",
        "How can one attain enlightenment according to Krishna?"
    ]
    
    k = 3
    results = []

    for q in questions:
        answer, retrieved_docs = rag_answer_local(q, k=k)
        print("\nQuestion:", q)
        print("Retrieved verses:\n", "\n".join(retrieved_docs))
        print("Answer:\n", answer)

        # BLEU
        reference = [doc.split() for doc in retrieved_docs]
        candidate = answer.split()
        smooth_fn = SmoothingFunction().method1
        bleu = sentence_bleu(reference, candidate, smoothing_function=smooth_fn)
        print("BLEUscore:", bleu)

        # BERTScore
        P_list, R_list, F1_list = [], [], []
        for ref in retrieved_docs:
            P, R, F1 = bert_score([answer], [ref], lang="en")
            P_list.append(P.item())
            R_list.append(R.item())
            F1_list.append(F1.item())
        bert_f1 = max(F1_list)
        print("BERTScore F1 (max over top-k):", bert_f1)

        results.append({
            "question": q,
            "generated_answer": answer,
            "reference_statements": " | ".join(retrieved_docs),
            "bleu_score": bleu,
            "bert_f1": bert_f1
        })
        print("="*80)

# ========================
# SAVE RESULTS
# ========================
df_results = pd.DataFrame(results)

# Save to PDF
pdf_file = "rag_evaluation_results_qwen2B.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
styles = getSampleStyleSheet()
elements = []

elements.append(Paragraph("RAG Evaluation Results - Qwen-2B", styles['Heading1']))
elements.append(Spacer(1, 12))

for _, row in df_results.iterrows():
    elements.append(Paragraph(f"Q: {row['question']}", styles['Heading3']))
    elements.append(Paragraph(f"A: {row['generated_answer']}", styles['Normal']))
    elements.append(Paragraph(f"References: {row['reference_statements']}", styles['Normal']))
    elements.append(Paragraph(f"BLEU: {row['bleu_score']:.4f}, BERT F1: {row['bert_f1']:.4f}", styles['Normal']))
    elements.append(Spacer(1, 12))

doc.build(elements)
print(f"Saved results to {pdf_file}")
