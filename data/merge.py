import pandas as pd
import json

# ---- File paths ----
CSV_PATH = "faq.csv"
JSON_PATH = "mental_health_qa.json"
OUTPUT_PATH = "final_merged.csv"

# ---- Load CSV ----
df = pd.read_csv(CSV_PATH)

# ---- Load JSON ----
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Handle both formats: list of objects OR dict of lists
if isinstance(data, list):
    instructions = [d["instruction"] for d in data]
    responses = [d["response"] for d in data]
else:
    instructions = data["instruction"]
    responses = data["response"]

# ---- MERGE with clean formatting ----
df["Questions"] = df["Questions"] + "\n\nInstruction:\n" + pd.Series(instructions)
df["Answers"]   = df["Answers"]   + "\n\nResponse:\n"   + pd.Series(responses)

# ---- Save clean CSV ----
df.to_csv(OUTPUT_PATH, index=False)

print("Clean merged CSV saved to:", OUTPUT_PATH)
