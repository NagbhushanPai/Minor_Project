import os
import sys
import logging
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# Logging: shares app.log with app.py so both apps' activity can be tailed
# from one place during a demo.
LOG_FILE = "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger("rag_qa_app.fast")

# Set page config
st.set_page_config(page_title="Fast Bhagavad Gita Q&A", page_icon="🕉️", layout="wide")

# Header
st.markdown("# 🕉️ Bhagavad Gita Q&A - Fast Version with AI Summary")
st.markdown("**Get instant verse search + AI-powered summaries**")

# Configuration
DATA_DIR = "./data"
INDEX_DIR = "./faiss_index_fast"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_fast_vectorstore():
    """Load or create a simple FAISS index with CSV data"""
    try:
        # Check if index exists
        if os.path.exists(INDEX_DIR):
            logger.info("Existing FAISS index found at %s. Loading...", INDEX_DIR)
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded successfully.")
            return vectorstore

        # Load CSV data
        csv_path = os.path.join(DATA_DIR, "bhagavad_gita_verses.csv")
        if not os.path.exists(csv_path):
            logger.error("CSV file not found at %s", csv_path)
            st.error("❌ CSV file not found!")
            return None

        df = pd.read_csv(csv_path)
        logger.info("Loading %d verses from CSV...", len(df))
        st.info(f"📊 Loading {len(df)} verses...")
        
        # Create simple documents
        docs = []
        for _, row in df.iterrows():
            content = f"Chapter {row['chapter_number']}: {row['chapter_title']}\n"
            content += f"Verse {row['chapter_verse']}: {row['translation']}"
            
            docs.append(Document(
                page_content=content,
                metadata={
                    "chapter": row['chapter_number'],
                    "verse": row['chapter_verse'],
                    "title": row['chapter_title']
                }
            ))
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Save for future use
        vectorstore.save_local(INDEX_DIR)
        logger.info("FAISS index created and saved to %s.", INDEX_DIR)
        st.success("✅ Index created!")

        return vectorstore

    except Exception as e:
        logger.exception("Error loading/creating fast vectorstore")
        st.error(f"❌ Error: {str(e)}")
        return None

@st.cache_resource
def load_summarizer():
    """Load a fast summarization model"""
    try:
        summarizer = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_new_tokens=100,
            do_sample=False,
            temperature=0.1,
            truncation=True
        )
        return summarizer
    except Exception as e:
        st.error(f"❌ Error loading summarizer: {str(e)}")
        return None

def simple_search(question, vectorstore):
    """Simple similarity search without LLM"""
    try:
        logger.info("Question received: %r", question)
        # Search for similar documents
        docs = vectorstore.similarity_search(question, k=3)
        logger.info("Search returned %d verse(s).", len(docs))
        return docs
    except Exception as e:
        logger.exception("Search error for question: %r", question)
        st.error(f"❌ Search error: {str(e)}")
        return []

def generate_summary(question, docs, summarizer):
    """Generate a quick summary based on the question and found verses"""
    try:
        if not docs or not summarizer:
            return None
        
        # Combine the relevant verses
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a prompt for summarization
        prompt = f"Question: {question}\n\nRelevant verses from Bhagavad Gita:\n{context}\n\nSummarize the key teachings related to the question:"
        
        # Generate summary
        result = summarizer(prompt)
        if result and len(result) > 0:
            return result[0]['generated_text'].strip()
        return None
        
    except Exception as e:
        st.error(f"❌ Summary error: {str(e)}")
        return None

def main():
    # Load vectorstore and summarizer
    with st.spinner("🔄 Loading verses and AI model..."):
        vectorstore = load_fast_vectorstore()
        summarizer = load_summarizer()
    
    if vectorstore is None:
        st.stop()
    
    # Show status
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Verses loaded - instant search ready!")
    with col2:
        if summarizer:
            st.success("✅ AI summarizer ready!")
        else:
            st.warning("⚠️ AI summarizer failed to load")
    
    # User input
    st.markdown("### 🤔 Ask about the Bhagavad Gita:")
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What does Krishna say about dharma?",
        key="question_input"
    )
    
    # Example questions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💭 About Dharma"):
            question = "What does Krishna say about dharma?"
    with col2:
        if st.button("⚔️ About Duty"):
            question = "What is duty according to Krishna?"
    with col3:
        if st.button("🧘 About Yoga"):
            question = "What is karma yoga?"
    
    if question:
        with st.spinner("🔍 Searching verses..."):
            docs = simple_search(question, vectorstore)
        
        if docs:
            # Create two columns for results
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### 📚 Relevant Verses:")
                for i, doc in enumerate(docs, 1):
                    with st.expander(f"📄 Verse {i} - Chapter {doc.metadata.get('chapter', '?')}"):
                        st.markdown(doc.page_content)
            
            with col2:
                if summarizer:
                    with st.spinner("🤖 Generating AI summary..."):
                        summary = generate_summary(question, docs, summarizer)
                    
                    st.markdown("### 💡 AI Summary:")
                    if summary:
                        st.markdown(f"""
                        <div style="background-color: rgba(50, 50, 50, 0.8); color: white; padding: 1rem; border-radius: 10px; border-left: 5px solid #00ff88; border: 1px solid #444;">
                        {summary}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Could not generate summary")
                else:
                    st.info("📝 AI summarizer not available - showing verses only")
                        
            # Summary of results
            chapters = set(doc.metadata.get('chapter', 'Unknown') for doc in docs)
            st.info(f"✨ Found {len(docs)} relevant verses from Chapter(s): {', '.join(map(str, sorted(chapters)))}")
        else:
            st.warning("❌ No relevant verses found. Try rephrasing your question.")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 How to use:
    1. **Type your question** in the text box above
    2. **Or click** one of the example buttons  
    3. **Get instant verse search** + **AI-powered summary**
    
    **Features**:
    - 🏎️ **Instant search**: Find relevant verses in seconds
    - 🤖 **AI Summary**: Get key teachings explained
    - 📊 **Complete data**: All 18 chapters with 641 verses
    """)

if __name__ == "__main__":
    main()
