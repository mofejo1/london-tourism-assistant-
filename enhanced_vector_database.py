# RETRIEVAL-AUGMENTED GENERATION (RAG) - LONDON TOURISM ASSISTANT
# Enhanced version for London Tourism PDFs with embedding download fixes
# ***

# Goals: 
# - Process London Tourism PDFs
# - Create vector database for RAG
# - Enable similarity search for tourism questions
# - Handle embedding model download issues

# LIBRARIES 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import pandas as pd
import yaml
import os
import fitz 
import shutil
import time
from pprint import pprint

# CLAUDE API SETUP - Following your credentials pattern
claude_api_key = yaml.safe_load(open("credentials.yml"))['claude']

# Clear corrupted cache function
def clear_model_cache(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Clear corrupted HuggingFace cache for a specific model"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    if os.path.exists(cache_dir):
        # Look for model-specific cache
        for item in os.listdir(cache_dir):
            if "all-MiniLM-L6-v2" in item:
                cache_path = os.path.join(cache_dir, item)
                print(f"Clearing cache: {cache_path}")
                try:
                    shutil.rmtree(cache_path)
                    print("✅ Cache cleared successfully")
                except Exception as e:
                    print(f"⚠️ Warning: Could not clear cache: {e}")

# Initialize embeddings with retry logic
def initialize_embeddings(max_retries=3):
    """Initialize HuggingFace embeddings with retry logic"""
    
    # Set environment variables
    os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Force online mode
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to load embedding model (attempt {attempt + 1}/{max_retries})...")
            
            embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            
            # Test the embedding function
            test_embedding = embedding_function.embed_query("test")
            print(f"✅ Embedding model loaded successfully! (dim: {len(test_embedding)})")
            return embedding_function
            
        except OSError as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            
            if "Consistency check failed" in str(e) or "should be of size" in str(e):
                print("Detected corrupted download. Clearing cache...")
                clear_model_cache()
                
                if attempt < max_retries - 1:
                    print("Waiting before retry...")
                    time.sleep(2)
            else:
                # For other errors, raise immediately
                raise
    
    # If all retries failed
    raise Exception("Failed to load embedding model after all retries")

# 1.0 LONDON TOURISM PDF PROCESSING ----

# Updated to use your PDF folder structure
pdf_folder_path = "/Users/p/Desktop/AIDatascience/Generative_Ai/ChatBot/pdfs"

# Check if folder exists
if not os.path.exists(pdf_folder_path):
    raise FileNotFoundError(f"PDF folder not found: {pdf_folder_path}")

# Get all PDF file paths
pdf_files = [os.path.join(pdf_folder_path, f) 
             for f in os.listdir(pdf_folder_path) 
             if f.endswith('.pdf')]

print(f"\nFound {len(pdf_files)} PDF files:")
for pdf in pdf_files:
    print(f"  - {os.path.basename(pdf)}")

# Load and combine all London tourism documents
all_documents = []

for pdf_file in pdf_files:
    print(f"\nProcessing: {os.path.basename(pdf_file)}")
    try:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
       
        # Extract meaningful title and set metadata
        pdf_title = os.path.splitext(os.path.basename(pdf_file))[0]
        pdf_author = "London Tourism Authority"  # More appropriate for tourism docs

        # Attach title and source info to each page's metadata
        for doc in docs:
            doc.metadata.setdefault("title", pdf_title)
            doc.metadata.setdefault("author", pdf_author)
            doc.metadata.setdefault("source_file", os.path.basename(pdf_file))
            doc.metadata.setdefault("document_type", "London Tourism Guide")

        all_documents.extend(docs)
        print(f"  ✅ Loaded {len(docs)} pages")
        
    except Exception as e:
        print(f"  ❌ Error processing {os.path.basename(pdf_file)}: {e}")

print(f"\nTotal documents loaded: {len(all_documents)}")

if not all_documents:
    raise ValueError("No documents were loaded. Please check your PDF files.")

# Preview one document
print("\n" + "="*50)
print("SAMPLE DOCUMENT PREVIEW:")
print("="*50)
print("Content:", all_documents[0].page_content[:300], "...")
print("\nMetadata:", all_documents[0].metadata)

# Check document lengths
print(f"\nDocument length statistics:")
lengths = [len(doc.page_content) for doc in all_documents]
print(f"  Average length: {sum(lengths)/len(lengths):.0f} characters")
print(f"  Min length: {min(lengths)} characters")
print(f"  Max length: {max(lengths)} characters")

# * Text Splitting - Tourism-optimized chunking
# Using smaller chunks for better tourism Q&A retrieval

CHUNK_SIZE = 1000  # Reduced for better tourism fact retrieval
CHUNK_OVERLAP = 200  # Good overlap for context

# Recursive Character Splitter: Uses "smart" splitting for tourism content
text_splitter_recursive = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],  # Added separators for better splitting
    length_function=len,
)

docs_recursive = text_splitter_recursive.split_documents(all_documents)

print(f"\nAfter splitting: {len(docs_recursive)} chunks")

# * Post Processing Text - Tourism Context Enhancement

# IMPORTANT: Prepend tourism context to each chunk
# - Helps with tourism-specific searches and citations
for doc in docs_recursive:
    # Retrieve metadata
    title = doc.metadata.get('title', 'Unknown Guide')
    author = doc.metadata.get('author', 'Unknown Author')
    source_file = doc.metadata.get('source_file', 'Unknown Source')
    doc_type = doc.metadata.get('document_type', 'Unknown Type')
    
    # Enhanced content for tourism assistant
    updated_content = f"Title: {title}\nSource: {source_file}\nType: {doc_type}\n\n{doc.page_content}"
    
    # Update the document's page content
    doc.page_content = updated_content

print("✅ Enhanced document chunks with tourism metadata")

# Sample enhanced chunk
print("\n" + "="*50)
print("SAMPLE ENHANCED CHUNK:")
print("="*50)
print(docs_recursive[0].page_content[:400], "...")

# * Text Embeddings - Using updated HuggingFace embeddings with error handling

print("\n" + "="*50)
print("INITIALIZING EMBEDDINGS:")
print("="*50)

# Initialize embeddings with retry logic
embedding_function = initialize_embeddings()

# * Langchain Vector Store: Chroma DB for London Tourism
# Following your pattern but with tourism-specific database

# Create London Tourism vector database
print("\n" + "="*50)
print("CREATING VECTOR DATABASE:")
print("="*50)

try:
    # Check if database already exists
    db_path = "data/london_tourism_db"
    
    if os.path.exists(db_path):
        print(f"Found existing database at {db_path}")
        print("Do you want to:")
        print("1. Use existing database")
        print("2. Recreate database from scratch")
        
        # For automated scripts, default to recreating
        # In interactive mode, you can add input() here
        choice = "2"  # Change to "1" to use existing
        
        if choice == "1":
            vectorstore = Chroma(
                embedding_function=embedding_function, 
                persist_directory=db_path
            )
            print("✅ Loaded existing London Tourism vector database!")
        else:
            # Remove existing database
            shutil.rmtree(db_path)
            print("Removed existing database. Creating new one...")
            
            vectorstore = Chroma.from_documents(
                docs_recursive, 
                embedding=embedding_function, 
                persist_directory=db_path
            )
            print("✅ New London Tourism vector database created!")
    else:
        # Create new database
        vectorstore = Chroma.from_documents(
            docs_recursive, 
            embedding=embedding_function, 
            persist_directory=db_path
        )
        print("✅ London Tourism vector database created!")
        
except Exception as e:
    print(f"❌ Error creating vector database: {e}")
    raise

# * Similarity Search: Testing with London tourism queries

print("\n" + "="*50)
print("TESTING TOURISM SIMILARITY SEARCH:")
print("="*50)

# Test queries for London tourism
test_queries = [
    "What are the best attractions to visit in London?",
    "Where can I find good restaurants in London?",
    "How do I get around London using public transport?",
    "What are some free things to do in London?",
    "Best museums to visit in London"
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    try:
        results = vectorstore.similarity_search(query, k=2)
        print(f"Found {len(results)} relevant chunks:")
        
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Source: {result.metadata.get('source_file', 'Unknown')}")
            print(f"    Title: {result.metadata.get('title', 'Unknown')}")
            print(f"    Content preview: {result.page_content[:150]}...")
            
    except Exception as e:
        print(f"  ❌ Error during search: {e}")

# Save configuration for later use
config = {
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "total_chunks": len(docs_recursive),
    "pdf_files": [os.path.basename(f) for f in pdf_files]
}

config_path = "data/london_tourism_config.yaml"
os.makedirs("data", exist_ok=True)
with open(config_path, 'w') as f:
    yaml.dump(config, f)

print("\n" + "="*50)
print("✅ LONDON TOURISM VECTOR DATABASE READY!")
print(f"Database location: {db_path}")
print(f"Configuration saved: {config_path}")
print(f"Total chunks: {len(docs_recursive)}")
print("Ready for RAG chatbot integration!")
print("="*50)