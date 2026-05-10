# app.py - Hybrid RAG + Groq LLM for TutorConnect Pakistan (with conversation memory)
from dotenv import load_dotenv
import os
import json
import sqlite3
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import json
from datetime import datetime


# -------------------- Configuration --------------------
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set. Please set it in .env or environment variables.")
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)
CORS(app)

# -------------------- In-Memory Conversation History --------------------
session_histories = {}          # { session_id: list of {"role": "user"/"assistant", "content": "..."} }
MAX_HISTORY_LEN = 10            # number of exchanges (each exchange = 2 messages) to keep

# -------------------- Database Setup --------------------
DB_PATH = "tutorconnect_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            user_message TEXT,
            bot_response TEXT,
            intent TEXT,
            confidence REAL,
            escalated BOOLEAN DEFAULT 0,
            timestamp TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            reason TEXT,
            handled BOOLEAN DEFAULT 0,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_interaction(session_id, user_msg, bot_resp, intent, confidence, escalated=False):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    interaction_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    c.execute('''
        INSERT INTO interactions (id, session_id, user_message, bot_response, intent, confidence, escalated, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (interaction_id, session_id, user_msg, bot_resp, intent, confidence, int(escalated), timestamp))
    conn.commit()
    conn.close()

def log_escalation(session_id, reason):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    escalation_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    c.execute('''
        INSERT INTO escalations (id, session_id, reason, handled, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (escalation_id, session_id, reason, 0, timestamp))
    conn.commit()
    conn.close()

# -------------------- RAG Setup --------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')
CHROMA_PATH = "./chroma_db"
os.makedirs(CHROMA_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection_name = "tutorconnect_docs"

def load_knowledge_base():
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        return json.load(f)

RESPONSES = load_knowledge_base()

def build_vector_store():
    try:
        coll = chroma_client.get_collection(collection_name)
        if coll.count() > 0:
            print("Vector store already exists, skipping rebuild.")
            return
    except:
        pass

    print("Building vector store from knowledge base...")
    collection = chroma_client.create_collection(name=collection_name) if collection_name not in [c.name for c in chroma_client.list_collections()] else chroma_client.get_collection(collection_name)
    
    chunks = []
    ids = []
    for idx, (key, text) in enumerate(RESPONSES.items()):
        chunks.append(text)
        ids.append(f"{key}_{idx}")
    
    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": key} for key in RESPONSES.keys()]
    )
    print(f"Vector store ready with {len(chunks)} documents.")

build_vector_store()

def retrieve_relevant_context(query, top_k=3):
    collection = chroma_client.get_collection(collection_name)
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    contexts = results['documents'][0] if results['documents'] else []
    return contexts

# -------------------- Groq Router --------------------
def route_query(query: str) -> str:
    """Return 'chat' for general conversation, 'rag' for platform questions."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a router. Classify the user's query as either 'chat' (for general conversation, greetings, or off-topic questions) or 'rag' (for questions specifically about tutors, tutoring, registration, subjects, areas, stipends, or the TutorConnect platform). Output only a JSON object: {'type': 'chat'} or {'type': 'rag'}."},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        classification = response.choices[0].message.content
        if '"type": "rag"' in classification:
            return "rag"
        else:
            return "chat"
    except Exception as e:
        print(f"Router error: {e}")
        domain_keywords = ["tutor", "registration", "subject", "area", "stipend", "donor", "match"]
        if any(kw in query.lower() for kw in domain_keywords):
            return "rag"
        return "chat"

# -------------------- Groq Conversation (General) with Memory --------------------
def chat_with_groq(query: str, session_id: str):
    # Retrieve history for this session
    history = session_histories.get(session_id, [])
    
    messages = [
        {"role": "system", "content": "You are a friendly assistant for TutorConnect Pakistan. Answer general questions and keep the conversation natural. If the user asks about something unrelated to tutoring, respond politely but redirect to the platform's purpose if possible."}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": query})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
    )
    answer = response.choices[0].message.content
    
    # Update history
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    # Trim to max exchanges
    if len(history) > MAX_HISTORY_LEN * 2:
        history = history[-MAX_HISTORY_LEN * 2:]
    session_histories[session_id] = history
    
    return answer

# -------------------- RAG with Groq Paraphrasing + Memory --------------------
def rag_with_groq(query: str, session_id: str):
    # Retrieve relevant documents
    contexts = retrieve_relevant_context(query, top_k=2)
    if not contexts:
        return "I'm sorry, I don't have enough information on that topic. Could you please rephrase or contact support?"
    
    context_str = "\n\n".join(contexts)
    
    # Include recent conversation history (last 4 messages = up to 2 exchanges)
    history = session_histories.get(session_id, [])
    history_text = ""
    if history:
        recent = history[-4:] if len(history) >= 4 else history
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
        history_text = f"### Previous conversation:\n{history_text}\n\n"
    
    prompt = f"""You are a helpful assistant for TutorConnect Pakistan, a platform connecting tutors with families.
{history_text}
Use the following information to answer the user's question.
If the information is not sufficient, say you don't know.
Do not repeat the information word for word – **paraphrase** it naturally in a clear, friendly, and helpful way.

### Information:
{context_str}

### User's Question:
{query}

### Your Answer (paraphrased, natural):"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = response.choices[0].message.content
    
    # Update history
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY_LEN * 2:
        history = history[-MAX_HISTORY_LEN * 2:]
    session_histories[session_id] = history
    
    return answer

# -------------------- Flask Endpoints --------------------
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message'}), 400
    user_message = data['message'].strip()
    session_id = data.get('session_id', str(uuid.uuid4()))
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    # Route the query
    query_type = route_query(user_message)
    
    # Generate response with memory
    if query_type == "chat":
        response_text = chat_with_groq(user_message, session_id)
        intent = "chat"
    else:
        response_text = rag_with_groq(user_message, session_id)
        intent = "rag_paraphrased"
    
    # Log interaction (database)
    log_interaction(session_id, user_message, response_text, intent, 0.9, False)
    
    return jsonify({
        'response': response_text,
        'intent': intent,
        'confidence': 0.9,
        'session_id': session_id,
        'escalated': False
    })

@app.route('/escalate', methods=['POST'])
def escalate():
    data = request.get_json()
    session_id = data.get('session_id', str(uuid.uuid4()))
    reason = data.get('reason', 'User requested human support')
    log_escalation(session_id, reason)
    log_interaction(session_id, "[ESCALATION REQUEST]", reason, "escalate_to_human", 1.0, True)
    return jsonify({
        'status': 'escalated',
        'message': 'A human support agent has been notified. They will contact you within 24 hours.',
        'session_id': session_id
    })

@app.route('/log', methods=['POST'])
def log_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing data'}), 400
    session_id = data.get('session_id', 'unknown')
    user_message = data.get('user_message', '')
    bot_response = data.get('bot_response', '')
    intent = data.get('intent', 'manual')
    confidence = data.get('confidence', 0.0)
    escalated = data.get('escalated', False)
    log_interaction(session_id, user_message, bot_response, intent, confidence, escalated)
    return jsonify({'status': 'logged'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'RAG+Groq', 'domain': 'TutorConnect Pakistan'})

# -------------------- Serve Static Frontend --------------------
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

# Paths for saving registrations
FAMILY_REGISTRATIONS_FILE = "family_registrations.json"
TUTOR_REGISTRATIONS_FILE = "tutor_registrations.json"

def save_registration_to_file(file_path, data):
    """Append a registration record to a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        records = []
    
    # Add timestamp
    data["registered_at"] = datetime.utcnow().isoformat()
    records.append(data)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

@app.route('/api/family-register', methods=['POST'])
def family_register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    # Optional: validation
    save_registration_to_file(FAMILY_REGISTRATIONS_FILE, data)
    return jsonify({"status": "saved", "message": "Family registration saved"}), 200

@app.route('/api/tutor-register', methods=['POST'])
def tutor_register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    save_registration_to_file(TUTOR_REGISTRATIONS_FILE, data)
    return jsonify({"status": "saved", "message": "Tutor registration saved"}), 200


# -------------------- Start Server --------------------
if __name__ == '__main__':
    print("🚀 Starting Hybrid RAG+Groq Chatbot for TutorConnect Pakistan...")
    print("📦 Embedding model: all-MiniLM-L6-v2")
    print("💾 Vector store: ChromaDB")
    print("🤖 Groq LLM: llama-3.3-70b-versatile")
    print("🧠 Conversation memory: enabled (up to 10 exchanges per session)")
    print("🌐 Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)