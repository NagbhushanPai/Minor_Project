import os
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from datasets import load_dataset
from langchain_core.documents import Document
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit page config
st.set_page_config(
    page_title="Q&A with Your Documents", 
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🤖 Q&A with Your Documents</h1></div>', unsafe_allow_html=True)
st.markdown("**Powered by RAG (Retrieval-Augmented Generation) with LangChain, FAISS, and Hugging Face**")

# Configuration
DATA_DIR = "./data"
INDEX_DIR = "./faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"  # Very fast, small model

@st.cache_resource(show_spinner="🔄 Indexing documents or loading FAISS index...")
def load_or_create_vectorstore():
    """
    Load existing FAISS index or create a new one from documents in the data directory or Hugging Face dataset.
    This function is cached to avoid reprocessing on every interaction.
    """
    try:
        # Check if FAISS index exists
        if os.path.exists(INDEX_DIR) and os.path.isdir(INDEX_DIR):
            st.info("📁 Loading existing FAISS index...")
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            return vectorstore

        # Check if data directory exists and has files
        docs = []
        if os.path.exists(DATA_DIR):
            # Load TXT files
            txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
            if txt_files:
                st.info(f"📚 Found {len(txt_files)} text files. Loading...")
                loader = DirectoryLoader(DATA_DIR, glob="*.txt")
                txt_docs = loader.load()
                docs.extend(txt_docs)
            
            # Load CSV files
            csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            if csv_files:
                st.info(f"📊 Found {len(csv_files)} CSV files. Loading...")
                for csv_file in csv_files:
                    csv_path = os.path.join(DATA_DIR, csv_file)
                    try:
                        df = pd.read_csv(csv_path)
                        st.info(f"📋 CSV columns: {list(df.columns)}")
                        
                        # Handle Bhagavad Gita CSV format
                        if 'translation' in df.columns and 'chapter_verse' in df.columns:
                            for _, row in df.iterrows():
                                content = f"Chapter: {row.get('chapter_title', 'Unknown')}\n"
                                content += f"Verse: {row.get('chapter_verse', 'Unknown')}\n"
                                content += f"Translation: {row.get('translation', '')}"
                                
                                docs.append(Document(
                                    page_content=content,
                                    metadata={
                                        "source": csv_file,
                                        "chapter": row.get('chapter_number', 'Unknown'),
                                        "verse": row.get('chapter_verse', 'Unknown')
                                    }
                                ))
                        # Generic CSV handling
                        else:
                            text_columns = [col for col in df.columns if df[col].dtype == 'object']
                            for idx, row in df.iterrows():
                                content = "\n".join([f"{col}: {row[col]}" for col in text_columns if pd.notna(row[col])])
                                if content.strip():
                                    docs.append(Document(
                                        page_content=content,
                                        metadata={"source": f"{csv_file}_row_{idx}"}
                                    ))
                        
                        st.info(f"📄 Loaded {len([d for d in docs if csv_file in d.metadata.get('source', '')])} documents from {csv_file}")
                    except Exception as e:
                        st.error(f"❌ Error loading CSV file {csv_file}: {str(e)}")
            
            if docs:
                st.success(f"✅ Total documents loaded from local files: {len(docs)}")
            else:
                st.warning("⚠️ No valid documents found in local files.")

        # If no .txt files, try loading from Hugging Face dataset
        if not docs:
            st.info("📦 No .txt files found. Loading Hugging Face dataset: om-ashish-soni/vivechan-spritual-text-dataset-v3 ...")
            try:
                ds = load_dataset("om-ashish-soni/vivechan-spritual-text-dataset-v3", split="train")
                
                # Check if dataset is empty
                if len(ds) == 0:
                    st.error("❌ The dataset is empty!")
                    return None
                
                # Check what fields are available in the dataset
                sample = ds[0]
                st.info(f"Dataset fields: {list(sample.keys())}")
                
                # Limit dataset size for faster processing (use only first 1000 documents)
                dataset_limit = min(1000, len(ds))  # Ensure we don't exceed dataset size
                st.info(f"📊 Using first {dataset_limit} documents from {len(ds)} total for faster processing...")
                
                # Try different possible text field names
                text_field = None
                for field in ['text', 'content', 'document', 'passage', 'body', 'description']:
                    if field in sample:
                        text_field = field
                        break
                
                if text_field:
                    # Only take first dataset_limit documents
                    try:
                        limited_ds = ds.select(range(dataset_limit))
                        docs = []
                        for i, ex in enumerate(limited_ds):
                            text_content = ex.get(text_field, "")
                            if text_content and len(str(text_content).strip()) > 50:
                                docs.append(Document(
                                    page_content=str(text_content), 
                                    metadata={"source": f"dataset_row_{i}"}
                                ))
                        st.info(f"📄 Loaded {len(docs)} documents from Hugging Face dataset using field '{text_field}'.")
                    except Exception as e:
                        st.error(f"❌ Error processing dataset documents: {str(e)}")
                        return None
                else:
                    st.error(f"❌ Could not find text field in dataset. Available fields: {list(sample.keys())}")
                    # Try to use the first string field as fallback
                    string_fields = [k for k, v in sample.items() if isinstance(v, str) and v.strip()]
                    if string_fields:
                        fallback_field = string_fields[0]
                        st.info(f"🔄 Trying fallback field: '{fallback_field}'")
                        try:
                            limited_ds = ds.select(range(dataset_limit))
                            docs = []
                            for i, ex in enumerate(limited_ds):
                                text_content = ex.get(fallback_field, "")
                                if text_content and len(str(text_content).strip()) > 20:
                                    docs.append(Document(
                                        page_content=str(text_content), 
                                        metadata={"source": f"dataset_row_{i}"}
                                    ))
                            st.info(f"📄 Loaded {len(docs)} documents using fallback field '{fallback_field}'.")
                        except Exception as e:
                            st.error(f"❌ Error with fallback field: {str(e)}")
                            return None
                    else:
                        return None
                    
            except Exception as e:
                st.error(f"❌ Error loading Hugging Face dataset: {str(e)}")
                return None

        if not docs:
            st.error("❌ No documents were loaded from either local files or Hugging Face dataset!")
            return None

        # Split documents into chunks (very small chunks for speed)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,  # Much smaller for faster processing
            chunk_overlap=20,  # Minimal overlap
            separators=["\n\n", "\n", " ", ""]
        )
        docs_chunks = splitter.split_documents(docs)
        st.info(f"✂️ Split into {len(docs_chunks)} chunks")

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(docs_chunks, embeddings)

        # Save FAISS index for future use
        vectorstore.save_local(INDEX_DIR)
        st.success("✅ FAISS index created and saved successfully!")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ Error loading/creating vectorstore: {str(e)}")
        return None

@st.cache_resource
def load_llm_pipeline():
    """
    Load the LLM pipeline. This is cached to avoid reloading the model.
    """
    try:
        llm_pipeline = pipeline(
            "text2text-generation",
            model=LLM_MODEL_NAME,
            tokenizer=LLM_MODEL_NAME,
            max_new_tokens=50,  # Very short answers for speed
            do_sample=False,  # Disable sampling for speed
            temperature=0.0,  # No randomness for speed
            truncation=True,  # Enable truncation
            device_map="auto",  # Use best available device
            torch_dtype="auto"  # Use optimal data type
        )
        return HuggingFacePipeline(pipeline=llm_pipeline)
    except Exception as e:
        st.error(f"❌ Error loading LLM: {str(e)}")
        st.info("💡 Try using a different model or check your transformers installation.")
        return None

def simple_keyword_search(question, vectorstore):
    """Simple keyword-based search for ultra-fast responses"""
    question_lower = question.lower()
    
    # Quick responses for common questions
    quick_answers = {
        "krishna": "Krishna is the Supreme Lord in the Bhagavad Gita, who serves as Arjun's charioteer and spiritual guide.",
        "arjun": "Arjun (Arjuna) is a great warrior prince and devotee who receives spiritual teachings from Krishna.",
        "dharma": "Dharma refers to righteous duty, moral law, and the natural order that sustains life and the universe.",
        "karma": "Karma means action and the law of cause and effect. Krishna teaches about performing duty without attachment to results.",
        "yoga": "Yoga means union or connection with the Divine. The Gita describes various paths like Karma Yoga, Bhakti Yoga, and Jnana Yoga.",
        "chapter 2": "Chapter 2 contains fundamental teachings about the soul, duty, and the nature of action without attachment."
    }
    
    for keyword, answer in quick_answers.items():
        if keyword in question_lower:
            return answer
    
    return None

def get_qa_chain(vectorstore, llm):
    """
    Create a RetrievalQA chain combining the vector store and LLM.
    """
    try:
        # Set up retriever (very few documents for speed)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1}  # Only 1 document for maximum speed
        )

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
        return qa_chain
    except Exception as e:
        st.error(f"❌ Error creating QA chain: {str(e)}")
        return None

def main():
    """
    Main application logic
    """
    # Sidebar with information
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This RAG (Retrieval-Augmented Generation) application allows you to:
        - Upload text documents
        - Ask questions about your documents
        - Get AI-powered answers with source references
        """)
        
        st.markdown("### 🔧 Configuration")
        st.code(f"""
Embedding Model: {EMBEDDING_MODEL_NAME}
LLM Model: {LLM_MODEL_NAME}
Chunk Size: 200 (optimized for speed)
Chunk Overlap: 20
Max Answer Length: 50 tokens
        """)
        
        st.markdown("### ⚡ Speed Mode")
        if st.button("🚀 Enable Ultra-Fast Mode"):
            st.session_state.fast_mode = True
            st.rerun()
        
        if st.button("🎯 Enable Quality Mode"):
            st.session_state.fast_mode = False
            st.rerun()
        
        st.markdown("### 📁 Data Directory")
        if os.path.exists(DATA_DIR):
            txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
            csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            
            if txt_files:
                st.success(f"✅ {len(txt_files)} .txt files found")
                for file in txt_files:
                    st.text(f"📄 {file}")
            
            if csv_files:
                st.success(f"✅ {len(csv_files)} .csv files found")
                for file in csv_files:
                    st.text(f"📊 {file}")
            
            if not txt_files and not csv_files:
                st.warning("⚠️ No .txt or .csv files found")
        else:
            st.error("❌ Data directory not found")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User input
        user_question = st.text_input(
            "🤔 Ask a question about your documents:",
            placeholder="e.g., What is the main topic discussed in the documents?"
        )
    
    with col2:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. Ask questions about the Bhagavad Gita
        2. Get AI-powered answers with sources
        3. Explore spiritual wisdom!
        
        **Try asking:**
        - "What is dharma?"
        - "Tell me about Chapter 2"
        - "What does Krishna teach?"
        """)

    # Load vectorstore and LLM
    with st.spinner("🔄 Loading models and data..."):
        vectorstore = load_or_create_vectorstore()
        llm = load_llm_pipeline()

    if vectorstore is None or llm is None:
        st.error("❌ Failed to initialize the application. Please check your setup.")
        return

    # Process user question
    if user_question:
        # Check for fast mode
        fast_mode = st.session_state.get('fast_mode', True)  # Default to fast
        
        if fast_mode:
            # Try quick keyword-based response first
            quick_answer = simple_keyword_search(user_question, vectorstore)
            if quick_answer:
                st.markdown("### ⚡ Quick Answer")
                st.markdown(f"""
                <div style="background-color: #e8f5e8; padding: 1rem; border-radius: 10px; border-left: 5px solid #28a745;">
                {quick_answer}
                </div>
                """, unsafe_allow_html=True)
                st.info("💡 This is a quick keyword-based answer. For detailed AI analysis, try Quality Mode.")
                return
        
        with st.spinner("🤔 Thinking..."):
            qa_chain = get_qa_chain(vectorstore, llm)
            
            if qa_chain is None:
                st.error("❌ Failed to create QA chain.")
                return
            
            try:
                # Add debug information
                st.info(f"🔍 Processing question: {user_question}")
                result = qa_chain({"query": user_question})
                
                # Check if we got a valid result
                if not result or not result.get('result'):
                    st.warning("⚠️ No answer generated. The model might be having issues.")
                    return
                
                # Display answer
                st.markdown("### 💡 Answer")
                answer = result['result'].strip()
                if answer:
                    st.markdown(f"""
                    <div style="background-color: #f0f7ff; padding: 1rem; border-radius: 10px; border-left: 5px solid #0066cc;">
                    {answer}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Empty answer received from the model.")
                
                # Display source documents
                if result.get("source_documents"):
                    st.markdown("### 📚 Source Documents")
                    
                    for i, doc in enumerate(result["source_documents"]):
                        with st.expander(f"📄 Source {i+1} - {doc.metadata.get('source', 'Unknown')}"):
                            content = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
                            st.text(content)
                            
                            # Show metadata if available
                            if doc.metadata:
                                st.markdown("**Metadata:**")
                                st.json(doc.metadata)
                else:
                    st.info("ℹ️ No source documents returned.")
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                st.info("💡 This might be due to model limitations or input length issues.")

    # Instructions
    st.markdown("---")
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📋 Instructions:
    1. **Add Documents**: Place your files in the `data/` folder:
       - `.txt` files: Plain text documents
       - `.csv` files: Structured data (automatic column detection)
    2. **First Run**: The app will automatically index your documents (this may take a few minutes)
    3. **Ask Questions**: Type your question in the input box above
    4. **View Results**: Get AI-powered answers with source document references
    
    **Supported File Types:**
    - 📄 **Text Files (.txt)**: Any plain text content
    - 📊 **CSV Files (.csv)**: Structured data with automatic field detection
    
    **Current Data**: Your CSV file contains Bhagavad Gita verses with chapter and translation information.
    
    **Note**: The FAISS index is saved locally, so subsequent runs will be much faster!
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
