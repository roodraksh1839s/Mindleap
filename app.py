<<<<<<< HEAD
from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pickle
import mysql.connector
from datetime import datetime

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
    app.run(debug=True)
=======
from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pickle
import mysql.connector
from datetime import datetime

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
    app.run(debug=True)
>>>>>>> ed3b1334f9cd78f66ced33a83d1942ef8cbd3f74
# print("DEBUG ROLE:", session.get("role"))