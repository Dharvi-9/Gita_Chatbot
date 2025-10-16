# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()  # loads variables from .env
openai_api_key = os.getenv("OPENAI_API_KEY")

# -------------------------------
# Initialize OpenAI LLM
# -------------------------------
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.2,
    max_tokens=500,
    openai_api_key=openai_api_key
)

# -------------------------------
# Import your chatbot logic
# -------------------------------
# Make sure smart_query() and extract_paragraph() are defined
from gita_chatbot_logic import smart_query, extract_paragraph

# -------------------------------
# Initialize FastAPI app
# -------------------------------
app = FastAPI(title="Bhagavad Gita QA API", version="1.0")

# -------------------------------
# Request body schema
# -------------------------------
class QueryRequest(BaseModel):
    query: str

# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def home():
    return {
        "message": "Namaste, seeker of wisdom! 🙏 I am Vedara 📖 — your Gita chatbot. "
                   "I am here to share the eternal teachings of the Bhagavad Gita, guiding you toward peace and clarity ✨"
    }

# -------------------------------
# Main POST query endpoint
# -------------------------------
@app.post("/ask")
def ask_gita(req: QueryRequest):
    full_response = smart_query(req.query)
    return {"response": full_response}

# -------------------------------
# Optional GET endpoint (for browser testing)
# -------------------------------
@app.get("/ask")
def ask_gita_get(query: str):
    full_response = smart_query(query)
    return {"response": full_response}

# -------------------------------
# Uvicorn entrypoint
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
