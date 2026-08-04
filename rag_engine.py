import os
import numpy as np
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try modern vector store & embedding imports
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    USE_HUGGINGFACE = True
except ImportError:
    USE_HUGGINGFACE = False

VECTOR_DB_DIR = "models/chroma_db"

def load_and_split_pdf(pdf_path="docs/compliance_policy_2026.pdf"):
    """Loads PDF using pypdf and splits into contextual chunks."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Document not found at {pdf_path}")
    
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    documents = [Document(page_content=full_text, metadata={"source": pdf_path})]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def initialize_vector_store(pdf_path="docs/compliance_policy_2026.pdf"):
    """Builds and persists the vector index with ChromaDB + HuggingFace Embeddings."""
    chunks = load_and_split_pdf(pdf_path)
    print(f"🧩 Split document into {len(chunks)} contextual chunks.")

    print("🧠 Initializing HuggingFace Embeddings ('sentence-transformers/all-MiniLM-L6-v2')...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 Indexing into ChromaDB Vector Store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print("✅ ChromaDB Vector Store successfully built!")
    return vector_store

def query_tfidf_fallback(query, pdf_path="docs/compliance_policy_2026.pdf", top_k=2):
    """Fallback TF-IDF Vector Search Engine if external network drops."""
    chunks = load_and_split_pdf(pdf_path)
    texts = [doc.page_content for doc in chunks]
    
    vectorizer = TfidfVectorizer().fit(texts + [query])
    tfidf_matrix = vectorizer.transform(texts)
    query_vector = vectorizer.transform([query])
    
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append((chunks[idx], float(scores[idx])))
    return results

def query_rag_engine(query, top_k=2):
    """Executes semantic search with automated fallback handling."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if os.path.exists(VECTOR_DB_DIR):
            vector_store = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
        else:
            vector_store = initialize_vector_store()
        
        results = vector_store.similarity_search_with_score(query, k=top_k)
        return results, "ChromaDB + HuggingFace Vectors"
    
    except Exception as e:
        print(f"⚠️ HuggingFace/Chroma connection issue ({e}). Engaging local TF-IDF vector engine...")
        results = query_tfidf_fallback(query, top_k=top_k)
        return results, "Local High-Speed TF-IDF Engine"

if __name__ == "__main__":
    print("🚀 Initializing Enterprise RAG Engine Test...")
    sample_query = "What is the penalty for failing to review high risk flags?"
    print(f"\n🔍 Querying: '{sample_query}'")
    
    results, engine_type = query_rag_engine(sample_query)
    print(f"\n⚡ Engine Utilized: {engine_type}")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Result {i} (Relevance Score: {score:.4f}) ---")
        print(f"Source Content:\n{doc.page_content}")