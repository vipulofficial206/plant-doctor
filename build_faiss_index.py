import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(r"C:\Users\vipul\plant doctor\.env")

KNOWLEDGE_BASE_FOLDER = r"C:\Users\vipul\plant doctor\KnowledgeBase"
FAISS_INDEX_PATH = os.path.join(KNOWLEDGE_BASE_FOLDER, "faiss_index")
EMBEDDINGS_MODEL = "models/embedding-001"

def build_index():
    docs = []
    for file in os.listdir(KNOWLEDGE_BASE_FOLDER):
        if file.endswith(".pdf"):
            path = os.path.join(KNOWLEDGE_BASE_FOLDER, file)
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} chunks created.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vs.save_local(FAISS_INDEX_PATH)
    print("🎉 FAISS index built and saved successfully!")

if __name__ == "__main__":
    build_index()
