# gita_chatbot_logic.py

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings  # for FAISS

import os

from langchain.chat_models import ChatOpenAI

# Read API key from environment variable (GitHub Secret or local .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# --------------------------
# 1. Download NLTK data
# --------------------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --------------------------
# 2. Load datasets
# --------------------------
verses_data = pd.read_json("gita_translation_data.json")
modern_data = pd.read_json("modern_teachings.json")

verses_data['text'] = verses_data['text'].astype(str).str.strip()
modern_data['text'] = modern_data['text'].astype(str).str.strip()
modern_data = modern_data.drop(columns=['source_refs'], errors='ignore')

verses_data = verses_data[~verses_data['text'].str.lower().isin(['', 'no changes needed'])]
verses_data = verses_data.dropna(subset=['text']).reset_index(drop=True)
modern_data = modern_data.reset_index(drop=True)

# --------------------------
# 3. Preprocessing functions
# --------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
custom_stopwords = ["thou", "thy", "thee", "shall"]
stop_words.update(custom_stopwords)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

verses_data['tokens'] = verses_data['text'].apply(preprocess_text)
modern_data['tokens'] = modern_data['text'].apply(preprocess_text)

verses_data['verses_str'] = verses_data['tokens'].apply(lambda x: ' '.join(x))
modern_data['modern_str'] = modern_data['tokens'].apply(lambda x: ' '.join(x))

# --------------------------
# 4. Create embeddings and FAISS retriever
# --------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

texts = list(verses_data['verses_str']) + list(modern_data['modern_str'])
metadatas = [{"type": "verse"} for _ in verses_data['verses_str']] + [{"type": "modern"} for _ in modern_data['modern_str']]

# Use HuggingFaceEmbeddings only for vector storage
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --------------------------
# 5. OpenAI LLM setup
# --------------------------
# Make sure to set your OpenAI API key as environment variable: export OPENAI_API_KEY="YOUR_KEY"
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_output_tokens=500,
)

# --------------------------
# 6. Helper functions for query processing
# --------------------------
def clean_query(query):
    q = query.strip().strip('"').strip("'")
    q = re.sub(r'[^\w\s?.!]', '', q)
    return q.lower()

def is_greeting(query):
    greetings = ["hi", "hello", "hey", "namaste", "good morning", "good evening", "bye"]
    return query.lower().strip() in greetings

def greeting_response():
    return "Namaste, seeker! 🌿📖 I’m Vedara — your playful Gita companion. Ask me anything, and I’ll share the timeless wisdom of the Bhagavad Gita in a heartbeat ⚡✨"

def detect_intent(query):
    q_lower = query.lower()
    if any(word in q_lower for word in ["summarize", "summary"]):
        return "summary"
    elif any(word in q_lower for word in ["example", "illustration"]):
        return "example"
    elif any(word in q_lower for word in ["explain", "explanation", "meaning of", "what does"]):
        return "explain"
    else:
        return "qa"

def is_sensitive(query):
    sensitive_keywords = ["LGBTQ", "politics", "controversy", "war", "fight", "battle",
        "religion", "caste", "society", "mother", "father", "violence",
        "attack", "offensive", "blame","god","gods","gender","sex","religions",
        "kill", "murder", "betray", "marriage", "family", "suicide", "hindu", "muslim"
    ]
    return any(word.lower() in query.lower() for word in sensitive_keywords)

def get_context(query, top_k=3, similarity_threshold=0.6):
    docs_with_scores = retriever.vectorstore.similarity_search_with_score(query, k=top_k)
    filtered_docs = [doc for doc, score in docs_with_scores if score >= similarity_threshold]
    context_text = "\n".join([doc.page_content for doc in filtered_docs]) if filtered_docs else ""
    return context_text, len(filtered_docs)

def extract_paragraph(full_response):
    paragraphs = [p.strip() for p in full_response.split("\n") if p.strip()]
    return paragraphs[-1] if paragraphs else full_response.strip()

# --------------------------
# 7. Smart query function
# --------------------------
def smart_query(query):
    query_clean = clean_query(query)

    if is_greeting(query_clean):
        return greeting_response()

    intent = detect_intent(query_clean)

    # Domain check
    gita_keywords = ["gita", "krishna", "arjuna", "dharma", "karma", "yoga", "battlefield", "bhagavad", "verse", "teachings"]
    if intent not in ["summary", "example", "explain"] and not any(k in query_clean for k in gita_keywords):
        return "Ah, seeker! 🌞📖 My insights bloom only from the Bhagavad Gita 🌿✨"

    context, count = get_context(query_clean)
    if count == 0:
        return "Ah, seeker! 🌞📖 My insights bloom only from the Bhagavad Gita 🌿✨"

    # Sensitive handling
    if is_sensitive(query_clean):
        prompt = f"""
        The following query may be sensitive. Respond in a neutral, educational, and respectful way using Bhagavad Gita teachings.
        Context: {context}
        User query: {query_clean}
        """
        return llm(prompt).content

    # Answer based on intent
    if intent == "summary":
        summary_prompt = PromptTemplate(
            input_variables=["context"],
            template="Summarize the following Gita teachings in simple words in about 150 words.:\n\n{context}"
        )
        summarization_chain = LLMChain(llm=llm, prompt=summary_prompt)
        return summarization_chain.run(context=context)
    elif intent == "example":
        prompt = f"Provide practical examples based on this Gita context in about 150 words.:\n{context}"
        return llm(prompt).content
    elif intent == "explain":
        prompt = f"You are Vedara, a friendly Gita chatbot. Provide a clear explanation based on the Bhagavad Gita teachings.\nContext: {context}\nUser query: {query_clean}"
        return llm(prompt).content

    # Default QA
    prompt = f"Answer based on Gita teachings in about 150 words.:\n{context}\nQuery: {query_clean}"
    return llm(prompt).content

__all__ = ["smart_query", "extract_paragraph"]
