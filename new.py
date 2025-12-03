import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

import pandas as pd

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('data/faq.csv')
df.columns = [c.strip().lower() for c in df.columns]

questions = df['questions'].tolist()
answers = df['answers'].tolist()

'''FOR-OPENAI'''
# def generate_response(user_input):
#     """Send prompt to GPT-4o with context"""
#     top_33matches = search_best_match(user_input)
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