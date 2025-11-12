from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pickle
import mysql.connector
from datetime import datetime
import os
'''
--------CHAT_BOT INTEGRATION (OPENAI)--------
'''
from flask import jsonify, request
# from openai import OpenAI
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
import requests

app = Flask(__name__)

app.secret_key = "mindleap_secret_key"


try:
    
    model = pickle.load(open('phq_model.pkl', 'rb'))
except Exception as e:
    print("Model not found or failed to load — using rule-based scoring.", e)
    model = None

def get_db_connection():
    return mysql.connector.connect(
        host = 'localhost',
        user = 'root',
        password = 'superb$1839S',
        database = 'mindleap'
    )


df = pd.read_csv('data/faq.csv')

# client = OpenAI(api_key="AIzaSyDsOxXjJpL3skc06Yo5d00SmLpcXQABxAE")
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')

#Ensure columns are clean

df.columns = [c.strip().lower() for c in df.columns]

questions = df['questions'].tolist()
answers = df['answers'].tolist()

#Compute embeddings

# Load or create FAISS embeddings
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
    
    
def search_best_match(user_query, top_k=2):
    """Find top-k most similar Q&A pairs"""
    query_emb = embed_model.encode([user_query])
    distances, indices = index.search(np.array(query_emb), top_k)
    results =[]
    for idx in indices[0]:
        results.append((questions[idx], answers[idx]))
    return results

# def generate_response(user_input):
#     """Send prompt to GPT-4o with context"""
#     top_matches = search_best_match(user_input)
#     context = "\n".join([f"Q: {q}\nA: {a}" for q,a in top_matches])

#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a multilingual assistant that only answers from the provided context."},
#             {"role": "user", "content": f"Context:\n{context}\n\nUser question: {user_input}\nGive a helpful answer in the same language as the question."}
#         ]
#     )
#     reply = completion.choices[0].message.content
#     return reply.strip()

def generate_response(user_input):
    """Generate reply using Google Gemini API"""
    GEMINI_API_KEY = "AIzaSyDPmpOMohKrH-IWjomBgb8X26h2Ei_zPmg"
    GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    # Retrieve top matches from your FAQ data
    top_matches = search_best_match(user_input)
    context = "\n".join([f"Q: {q}\nA: {a}" for q, a in top_matches])

    # Prepare prompt
    prompt = (
        f"You are a multilingual, calm, and empathetic mental wellness assistant.\n"
        f"Use the given Q&A context to answer naturally in the same language as the user's question.\n\n"
        f"Context:\n{context}\n\nUser question: {user_input}"
    )

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        res = requests.post(GEMINI_URL, json=payload)
        data = res.json()

        # Extract response text safely
        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            print("⚠️ Gemini API unexpected response:", data)
            return "Sorry, I couldn’t find an appropriate answer right now."
    except Exception as e:
        print("⚠️ Gemini API Error:", e)
        # Fallback to FAQ answer directly
        return top_matches[0][1] if top_matches else "Sorry, I'm unable to respond right now."


@app.route('/')
def home():
    return render_template('Login.html')



@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    role = request.form.get('role')
    
    print("DEBUG LOGIN FORM:", email, password, role)  # 👈
    
    role = role.upper() if role else None

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM USER WHERE EMAIL=%s AND PASSWORD=%s AND ROLE=%s", (email, password, role))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    print("DEBUG USER FROM DB:", user)  # 👈

    
    
    if user:
        session['email'] = user['EMAIL']
        session['role'] = user['ROLE']  
        
        if user['ROLE'] == 'STUDENT':
            return redirect(url_for('student_ui'))
        elif user['ROLE'] == 'ADMIN':
            return redirect(url_for('admin'))
        
    else:
        flash("Invalid credntials or role")
        return redirect(url_for('home'))
    


@app.route('/student')
def student_ui():
    print("DEBUG SESSION:", session)   # 👈 See what’s stored
    print("ROLE IN SESSION:", session.get("role"))
    if 'role' in session and session['role'] == 'STUDENT':
        return render_template('UserUI.html')
    return redirect(url_for('home'))


@app.route('/aichat')
def chatbot_ui():
    return render_template("chatbot.html")

@app.route('/chat',methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message','').strip()
    
    if not user_msg:
        return jsonify({"reply":"Please write your query."})
    
    try:
        reply = generate_response(user_msg)
        return jsonify({'reply': reply})
    except Exception as e:
        print("Error:", e)
        top_match = search_best_match(user_msg)[0][1]
        return top_match or "Sorry, I couldn't process your request."
@app.route('/peer-support')
def peer_support_ui():
    return render_template('Peer_support.html')

@app.route('/resources')
def resources_ui():
    return render_template('resource.html')

@app.route('/book-session')
def booking_ui():
    return render_template('booking_page.html')

@app.route('/admin')
def admin():
    if 'role' not in session or session["role"] != 'ADMIN':
        return redirect(url_for('home'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM assessments ORDER BY SubmittedAt DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # 🔎 Debug: print what keys MySQL is returning
    if rows:
        print("Column keys from DB:", rows[0].keys())
    else:
        print("No rows found in assessments table")

    labels = ["minimal", "mild", "moderate", "moderately_severe", "severe"]
    counts = {label: 0 for label in labels}

    for row in rows:
        # prediction = row.get('Prediction') or row.get('prediction') or row.get('PREDICTION')
        prediction = row.get('PREDICTION')
        if prediction in counts:
            counts[prediction] += 1

    return render_template("newadmin.html", rows=rows, counts=counts)



@app.route('/form')
def form():
    return render_template('style.html')

# @app.route('/submit', methods=['POST'])
@app.route('/submit', methods=['POST'])
def submit():
    global model  # make sure we use the model defined at the top

    # collect integers from form
    try:
        responses = [int(request.form.get(f'q{i}', 0)) for i in range(1,10)]
    except:
        responses = []
        for i in range(1,10):
            val = request.form.get(f"q{i}", "0")
            try:
                responses.append(int(val))
            except:
                responses.append(0)

    total_score = int(sum(responses))

    risk_label = None

    # if you loaded a model and want to use it
    if model is not None:
        try:
            X = np.array(responses).reshape(1, -1)
            pred = model.predict(X)[0]
            risk_label = str(pred)
        except Exception as e:
            print("Model predict failed, falling back to rule-based:", e)
            model = None  # safe to disable the model

    if risk_label is None:  # fallback rule-based scoring
        if total_score <= 4:
            risk_label = "minimal"
        elif total_score <= 9:
            risk_label = "mild"
        elif total_score <= 14:
            risk_label = "moderate"
        elif total_score <= 19:
            risk_label = "moderately_severe"
        else:
            risk_label = "severe"
            
            
    # after risk_label is decided
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO assessments 
        (Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,PREDICTION,SubmittedAt)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    cursor.execute(query, (*responses, risk_label, datetime.now()))
    conn.commit()
    cursor.close()
    conn.close()


    # conn = get_db_connection()
    # cursor = conn.cursor()
    
    # query = """
    #     INSERT INTO ASSESSMENTS
    #     (Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,PREDICTION)
    #     VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        
    # cursor.execute(query, (*responses, risk_label))
    # conn.commit()
    # cursor.close()
    # conn.close()
    
    
    
    return render_template("result.html", score=total_score, risk=risk_label)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
# print("DEBUG ROLE:", session.get("role"))