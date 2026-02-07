import os
import json
import uvicorn
import numpy as np
import keras
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# FastAPI and Pydantic
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain and Google GenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Configuration & Globals
# -----------------------------
# Load environment variables (Adapt the path to your .env file)
load_dotenv(dotenv_path=r"C:\Users\vipul\plant doctor\.env")

# Paths and Models
# NOTE: Ensure these paths are correct for the environment where you run the API.
KNOWLEDGE_BASE_FOLDER = r"C:\Users\vipul\plant doctor\KnowledgeBase"
FAISS_INDEX_SUBDIR = "faiss_index"
EMBEDDINGS_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.5-pro"
TEMPERATURE = 0.2

# Globals for state management
vector_store_instance = None
llm_instance = None

# -----------------------------
# Load Image Classification Model
# -----------------------------
# Ensure the path to your Keras model is correct
Model = keras.models.load_model(r".\saved_models\1.keras")
class_names = [
    'American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm',
    'Becterial Blight in Rice', 'Brownspot', 'Common_Rust', 'Cotton Aphid',
    'Flag Smut', 'Gray_Leaf_Spot', 'Healthy Maize', 'Healthy Wheat',
    'Healthy cotton', 'Leaf Curl', 'Leaf smut', 'Mosaic sugarcane',
    'RedRot sugarcane', 'RedRust sugarcane', 'Rice Blast', 'Sugarcane Healthy',
    'Tungro', 'Wheat Brown leaf Rust', 'Wheat Stem fly', 'Wheat aphid',
    'Wheat black rust', 'Wheat leaf blight', 'Wheat mite', 'Wheat powdery mildew',
    'Wheat scab', 'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane',
    'bacterial_blight in Cotton', 'bollworm on Cotton', 'cotton mealy bug',
    'cotton whitefly', 'maize ear rot', 'maize fall armyworm', 'maize stem borer',
    'pink bollworm in cotton', 'red cotton bug', 'thirps on  cotton'
]

# -----------------------------
# RAG Helper Functions (Index/Chatbot)
# -----------------------------

def load_pdfs_from_folder(folder_path):
    all_documents = []
    if not os.path.exists(folder_path):
        print(f"Error: Knowledge base folder '{folder_path}' not found.")
        return []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            try:
                loader = PyPDFLoader(pdf_path)
                documents_from_pdf = loader.load()
                all_documents.extend(documents_from_pdf)
            except Exception as e:
                print(f"Error loading '{filename}': {e}")
    return all_documents


def split_documents_into_chunks(documents):
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
    )
    return splitter.split_documents(documents)


def get_or_create_vectorstore(knowledge_base_folder, embeddings_model=EMBEDDINGS_MODEL):
    """Loads an existing FAISS index or creates a new one from PDFs (the build_index logic)."""
    vector_store_path = os.path.join(knowledge_base_folder, FAISS_INDEX_SUBDIR)
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in environment variables!")

    embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model, google_api_key=google_api_key)
    
    faiss_index_file = os.path.join(vector_store_path, "index.faiss")
    faiss_pkl_file = os.path.join(vector_store_path, "index.pkl")

    # Attempt to load the existing index
    if os.path.exists(vector_store_path) and os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file):
        try:
            print("Loading existing FAISS index...")
            return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Warning: Error loading vector store: {e}. Rebuilding...")

    # Build new index if loading fails or index doesn't exist
    print("Building new FAISS index...")
    documents = load_pdfs_from_folder(knowledge_base_folder)
    chunks = split_documents_into_chunks(documents)
    
    if not chunks:
        print("Error: No document chunks found. Cannot build index.")
        return None

    print(f"✅ {len(chunks)} chunks created.")
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(vector_store_path, exist_ok=True)
    vector_store.save_local(vector_store_path)
    print("🎉 FAISS index built and saved successfully!")
    return vector_store

def ask_question(query_text: str):
    """Ask Gemini model a question using RAG and return plain text (now only for disease queries)."""
    if vector_store_instance is None or llm_instance is None:
        raise RuntimeError("Vector store or LLM not initialized.")

    docs = vector_store_instance.similarity_search(query_text, k=5)
    context_text = "\n".join([doc.page_content for doc in docs]) if docs else ""

    prompt = f"""
You are a Plant Pathology Assistant for farmers.
Answer in short, clear, practical language. Avoid jargon.

Context:
{context_text}

Question:
{query_text}

Answer:
"""
    response = llm_instance.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
# The rest of your FastAPI script remains the same.

def get_structured_disease_info(query_text: str):
    if llm_instance is None:
        raise RuntimeError("LLM not initialized.")

    triage_prompt = f"""
    Classify the following user query as 'GREETING' or 'DISEASE_QUERY':
    "{query_text}"
    """

    triage_response = llm_instance.invoke(triage_prompt, temperature=0.0)
    triage_class = triage_response.content.strip().upper()

    # Handle greetings conversationally
    if "GREETING" in triage_class:
        message = "👋 Hello there! I'm your friendly Plant Doctor. I can identify plant diseases and tell you their symptoms, causes, and treatments. What plant issue are you seeing today?"
        return {
            "type": "GREETING",
            "message": message,
            "chat_response": message
        }

    # Otherwise handle as disease info
    disease_name = query_text

    def safe_ask(q):
        try:
            ans = ask_question(q)
            return ans.strip()
        except Exception:
            return f"Could not find information for {disease_name}."

    symptoms = safe_ask(f"What are the symptoms of {disease_name}?")
    causes = safe_ask(f"What causes {disease_name}?")
    prevention = safe_ask(f"How can farmers prevent {disease_name}?")
    treatment = safe_ask(f"What are the treatments for {disease_name}?")

    chatbot_message = f"🩺 Here’s what I found about **{disease_name}**:\n\n" \
                      f"**Symptoms:** {symptoms}\n\n" \
                      f"**Causes:** {causes}\n\n" \
                      f"**Prevention:** {prevention}\n\n" \
                      f"**Treatment:** {treatment}\n\n" \
                      "Would you like me to suggest any organic remedies?"

    return {
        "type": "DISEASE_INFO",
        "symptoms": symptoms,
        "causes": causes,
        "prevention": prevention,
        "treatment": treatment,
        "chat_response": chatbot_message
    }

# -----------------------------
# Image Helper Functions
# -----------------------------

def read_file_as_image(data) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)))

def predict_image(image: np.ndarray):
    img_resized = np.array(Image.fromarray(image).resize((128, 128)))
    if img_resized.ndim == 2:
        img_resized = np.stack((img_resized,) * 3, axis=-1)
    elif img_resized.shape[-1] == 4:
        img_resized = img_resized[..., :3]

    img_batch = np.expand_dims(img_resized, 0).astype("float32")
    # Keras predict is being called
    prediction = Model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))
    return predicted_class, confidence

# -----------------------------
# FastAPI Lifespan
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup (loading models/index) and shutdown."""
    global vector_store_instance, llm_instance
    print("Starting up Plant Doctor API...")
    try:
        # Load or create the RAG vector store
        vector_store_instance = get_or_create_vectorstore(KNOWLEDGE_BASE_FOLDER)
        # Initialize the LLM
        llm_instance = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=TEMPERATURE)
        print("Initialization successful. Ready to serve requests.")
    except Exception as e:
        print(f"Startup error: {e}")
        raise RuntimeError(f"Failed to initialize server: {e}")
    yield
    print("Shutting down Plant Doctor API...")

# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(title="Plant Doctor API", version="2.1.0", lifespan=lifespan)

# CORS Middleware setup
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------

class DiseaseQuery(BaseModel):
    disease_name: str

# -----------------------------
# Endpoints
# -----------------------------

@app.get("/")
async def root():
    return {"message": "Plant Doctor API is running!", "docs_url": "/docs"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded image."""
    image = read_file_as_image(await file.read())
    predicted_class, confidence = predict_image(image)
    return {"predicted_class": predicted_class, "confidence": confidence}


@app.post("/analyze_disease_from_image")
async def analyze_disease(file: UploadFile = File(...)):
    """Predict disease from image, then fetch structured info."""
    if vector_store_instance is None or llm_instance is None:
        raise HTTPException(status_code=500, detail="Server not initialized.")

    image = read_file_as_image(await file.read())
    predicted_class, confidence = predict_image(image)

    print(f"Detected: {predicted_class} (conf: {confidence:.2f})")

    # Use the RAG system to get structured info
    disease_info = get_structured_disease_info(predicted_class)

    # Response (JSON format unchanged)
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "chatbot_answer": disease_info
    }


@app.post("/chatbot_disease_info")
async def chatbot_disease_info(query: DiseaseQuery):
    if vector_store_instance is None or llm_instance is None:
        raise HTTPException(status_code=500, detail="Server not initialized.")
        
    if not query.disease_name:
        raise HTTPException(status_code=400, detail="Disease name is required.")

    print(f"Chatbot request: {query.disease_name}")
    
    disease_info = get_structured_disease_info(query.disease_name)

    if disease_info["type"] == "GREETING":
        return {
            "query_type": "greeting",
            "chatbot_message": disease_info["chat_response"]
        }

    return {
        "query_type": "disease_info",
        "disease_name": query.disease_name,
        "chatbot_message": disease_info["chat_response"],
        "structured_data": {
            "symptoms": disease_info["symptoms"],
            "causes": disease_info["causes"],
            "prevention": disease_info["prevention"],
            "treatment": disease_info["treatment"]
        }
    }

# -----------------------------
# Run app
# -----------------------------

if __name__ == "__main__":
    # Runs the FastAPI application
    uvicorn.run(app, host="localhost", port=9087)