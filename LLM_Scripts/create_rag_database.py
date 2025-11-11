
import os
import glob
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader

# --- Configuration ---
SOURCE_DOCUMENTS_PATH = "rag_data_sources"
DB_FAISS_PATH = "faiss_index_original" # A new name to avoid conflict
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def main():
    print(f"--- Creating Original RAG Database from '{SOURCE_DOCUMENTS_PATH}' ---")
    
    # This will find both CSVs and PDFs
    all_files = glob.glob(os.path.join(SOURCE_DOCUMENTS_PATH, "*"))
    
    documents = []
    print("\n--- Loading all documents ---")
    for file_path in all_files:
        print(f"  > Loading: {os.path.basename(file_path)}")
        try:
            if file_path.endswith(".pdf"):
                # This will fail silently if pypdf is not installed, which is what happened before
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith(".csv"):
                # Use a simple loader without specifying a source column
                loader = CSVLoader(file_path=file_path)
                documents.extend(loader.load())
        except Exception as e:
            # The original script likely ignored errors
            print(f"    WARNING: Failed to load {file_path}. Error: {e}")
            
    if not documents:
        raise ValueError("No documents were successfully loaded.")

    print(f"\n--- Splitting documents... ---")
    texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    print(f"--- Split content into {len(texts)} searchable chunks. ---")

    print(f"\n--- Loading embedding model '{EMBEDDING_MODEL}'... ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    
    print("\n--- Creating the FAISS vector database... ---")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"\n--- SUCCESS! Original database recreated at '{DB_FAISS_PATH}' ---")

if __name__ == "__main__":
    main()