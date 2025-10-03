import os
import torch
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========================
# NLTK SETUP
# ========================
nltk.download('punkt')

# ========================
# CONFIGURATION
# ========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Hugging Face Qwen model (local)
model_name = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",     # automatically uses MPS if available
    dtype=torch.float16    # avoids deprecation warning
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
print("Encoding documents...")
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
# RAG FUNCTION (Local Qwen)
# ========================

def rag_answer_online(question, k=3, max_new_tokens=100):
    # Retrieve top-k documents
    retrieved_docs = retrieve(question, k)
    context = "\n".join(retrieved_docs)
    
    prompt = f"Answer the question using ONLY the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
    
    # Tokenize and generate locally
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
    k = 3  # top-k retrieved verses
    results = []

    for q in questions:
        answer, retrieved_docs = rag_answer_online(q, k=k)
        print("\nQuestion:", q)
        print("Retrieved verses (context):\n", "\n".join(retrieved_docs))
        print("Answer:\n", answer)

        # BLEU SCORE
        reference = [doc.split() for doc in retrieved_docs]
        candidate = answer.split()
        smooth_fn = SmoothingFunction().method1
        bleu = sentence_bleu(reference, candidate, smoothing_function=smooth_fn)
        print("BLEUscore:", bleu)

        # BERT SCORE
        P_list, R_list, F1_list = [], [], []
        for ref in retrieved_docs:
            P, R, F1 = bert_score([answer], [ref], lang="en")
            P_list.append(P.item())
            R_list.append(R.item())
            F1_list.append(F1.item())
        bert_f1 = max(F1_list)
        print("BERTScore F1 (max over top-k):", bert_f1)

        # Save results for CSV
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

# Save to PDF (paragraph style)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

pdf_file = "rag_evaluation_results_qwen_local.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
styles = getSampleStyleSheet()
elements = []

# Title
elements.append(Paragraph("RAG(QWEN2_0.5B) Evaluation Results - Qwen Local", styles['Heading1']))
elements.append(Spacer(1, 12))  # small gap

# Add each Q/A block
for _, row in df_results.iterrows():
    elements.append(Paragraph(f"Q: {row['question']}", styles['Heading3']))
    elements.append(Paragraph(f"A: {row['generated_answer']}", styles['Normal']))
    elements.append(Paragraph(f"References: {row['reference_statements']}", styles['Normal']))
    elements.append(Paragraph(f"BLEU: {row['bleu_score']:.4f}, BERT F1: {row['bert_f1']:.4f}", styles['Normal']))
    elements.append(Spacer(1, 12))  # gap between questions

# Build PDF
doc.build(elements)
print(f"Saved results to {pdf_file}")
