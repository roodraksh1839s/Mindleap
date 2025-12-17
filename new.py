# =========================
# Imports
# =========================
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import numpy as np
import pickle
import mysql.connector
from datetime import datetime
import os
import pandas as pd
import requests
import faiss
from sentence_transformers import SentenceTransformer

# =========================
# Flask App Config
# =========================
app = Flask(__name__)
app.secret_key = "mindleap_secret_key"

# =========================
# ML Model Loading
# =========================
try:
    model = pickle.load(open('phq_model.pkl', 'rb'))
except Exception as e:
    print("Model not found or failed to load — using rule-based scoring.", e)
    model = None

# =========================
# Database
# =========================
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='superb$1839S',
        database='mindleap'
    )

# =========================
# cHAt_bot LOGS
# =========================
def save_chat_logs(user_email, user_message, bot_reply):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO CHAT_LOGS (USER_EMAIL, USER_MESSAGE, BOT_REPLY, CTREATED_AT)
        VALUES (%s,%s,%s,%s)
        """
        cursor.execute(
            query,
            (user_email,user_message, bot_reply, datetime.now())
        )
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(" Failed to save chat log:",e)
        
        
# =========================
# FAQ + Embeddings Setup
# =========================
df = pd.read_csv('data/faq.csv')
df.columns = [c.strip().lower() for c in df.columns]

questions = df['questions'].tolist()
answers = df['answers'].tolist()

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists("data/embeddings.pkl"):
    with open("data/embeddings.pkl", "rb") as f:
        embeddings, index = pickle.load(f)
    print("✅ Loaded cached embeddings")
else:
    embeddings = embed_model.encode(questions, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    with open("data/embeddings.pkl", "wb") as f:
        pickle.dump((embeddings, index), f)
    print("✅ Created and saved embeddings")

# =========================
# NLP Utilities
# =========================
def search_best_match(user_query, top_k=2):
    query_emb = embed_model.encode([user_query])
    distances, indices = index.search(np.array(query_emb), top_k)
    results = []
    for idx in indices[0]:
        results.append((questions[idx], answers[idx]))
    return results


def generate_response(user_input):
    GEMINI_API_KEY = "AIzaSyDPmpOMohKrH-IWjomBgb8X26h2Ei_zPmg"
    GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    top_matches = search_best_match(user_input)
    context = "\n".join([f"Q: {q}\nA: {a}" for q, a in top_matches])

    prompt = (
        "You are a multilingual, calm, and empathetic mental wellness assistant.\n"
        "Use the given Q&A context to answer naturally in the same language as the user's question.\n\n"
        "Use short sentences until user asks you to give in detail."
        f"Context:\n{context}\n\nUser question: {user_input}"
    )

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        res = requests.post(GEMINI_URL, json=payload)
        data = res.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        return "Sorry, I couldn’t find an appropriate answer right now."
    except Exception as e:
        print("⚠️ Gemini API Error:", e)
        return top_matches[0][1] if top_matches else "Sorry, I'm unable to respond right now."

# =========================
# Routes
# =========================
@app.route('/')
def home():
    return render_template('Login.html')


@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    role = request.form.get('role')
    role = role.upper() if role else None

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM USER WHERE EMAIL=%s AND PASSWORD=%s AND ROLE=%s",
        (email, password, role)
    )
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if user:
        session['email'] = user['EMAIL']
        session['role'] = user['ROLE']
        return redirect(url_for('student_ui' if user['ROLE'] == 'STUDENT' else 'admin'))

    flash("Invalid credntials or role")
    return redirect(url_for('home'))


@app.route('/student')
def student_ui():
    if 'role' in session and session['role'] == 'STUDENT':
        return render_template('UserUI.html')
    return redirect(url_for('home'))


@app.route('/aichat')
def chatbot_ui():
    return render_template("chatbot.html")


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({"reply": "Please write your query."})
    
    reply = generate_response(user_msg)
    
    # Get logged-in user (if available)
    user_email = session.get("email", "anonymous")
    
    # Save chat
    save_chat_logs(user_email, user_msg, reply)
    return jsonify({'reply': reply})


@app.route('/admin')
def admin():
    if session.get("role") != 'ADMIN':
        return redirect(url_for('home'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM assessments ORDER BY SubmittedAt DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    labels = ["minimal", "mild", "moderate", "moderately_severe", "severe"]
    counts = {label: 0 for label in labels}
    for row in rows:
        if row.get('PREDICTION') in counts:
            counts[row['PREDICTION']] += 1

    return render_template("Admin.html", rows=rows, counts=counts)

# =========================
# App Runner
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
