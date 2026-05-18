import os
import csv
import threading
from datetime import datetime

import joblib
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FEEDBACK_FILE        = "feedback.csv"
FEEDBACK_SIM_THRESHOLD = 0.92     # cosine similarity to count as "same article"
TOP_SHAP_FEATURES    = 10

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
model      = joblib.load("saved_models/model.pkl")
vectorizer = joblib.load("saved_models/vectorizer.pkl")
background = joblib.load("saved_models/background.pkl")

# LinearExplainer expects dense background
explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")

feature_names = vectorizer.get_feature_names_out()

# Verify label direction once at startup
print(f"[STARTUP] model.classes_ = {model.classes_}")
print(f"[STARTUP] 0 → FAKE  |  1 → REAL")

# ─────────────────────────────────────────────
# FEEDBACK STORE
# ─────────────────────────────────────────────
_feedback_lock = threading.Lock()

def _init_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "text", "predicted_label", "correct_label"])

def _save_feedback(text: str, predicted_label: str, correct_label: str):
    with _feedback_lock:
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), text, predicted_label, correct_label])

def _load_feedback() -> list[dict]:
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with _feedback_lock:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

_init_feedback()

# ─────────────────────────────────────────────
# FEEDBACK PRIORITY CHECK
# ─────────────────────────────────────────────
def check_feedback_priority(text: str) -> str | None:
    """
    Returns corrected label string ('FAKE'/'REAL') if a near-identical
    article exists in feedback.csv, else None.
    """
    feedback = _load_feedback()
    if not feedback:
        return None

    feedback_texts = [row["text"] for row in feedback]
    all_vecs = vectorizer.transform(feedback_texts + [text])

    input_vec     = all_vecs[-1]
    feedback_vecs = all_vecs[:-1]

    sims    = cosine_similarity(input_vec, feedback_vecs)[0]
    best_i  = int(sims.argmax())

    if sims[best_i] >= FEEDBACK_SIM_THRESHOLD:
        matched_label = feedback[best_i]["correct_label"].upper()
        print(f"[FEEDBACK HIT] sim={sims[best_i]:.3f} → label={matched_label}")
        return matched_label

    return None

# ─────────────────────────────────────────────
# SHAP EXPLANATION HELPER
# ─────────────────────────────────────────────
def build_shap_explanation(vector) -> list[dict]:
    shap_values = explainer(vector)
    values      = shap_values.values[0]

    explanation = [
        {"word": feature_names[i], "importance": float(values[i])}
        for i in range(len(values))
        if abs(values[i]) > 0.01
    ]
    explanation.sort(key=lambda x: abs(x["importance"]), reverse=True)
    return explanation[:TOP_SHAP_FEATURES]

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(title="Fake News Detector", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────
class NewsRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    text: str
    predicted_label: str       # what the model said ("FAKE" or "REAL")

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(news: NewsRequest):
    text   = news.text.strip()
    vector = vectorizer.transform([text])

    # ── 1. Check feedback priority first ─────
    feedback_label = check_feedback_priority(text)
    if feedback_label:
        return {
            "prediction": feedback_label,
            "confidence": 1.0,
            "source":     "feedback",          # frontend can badge this
            "shap":       build_shap_explanation(vector),
        }

    # ── 2. Model prediction ───────────────────
    prediction   = model.predict(vector)[0]          # 0 = FAKE, 1 = REAL
    proba        = model.predict_proba(vector)[0]    # [P(FAKE), P(REAL)]
    label        = "REAL" if prediction == 1 else "FAKE"
    confidence   = float(proba[prediction])          # probability of predicted class

    return {
        "prediction": label,
        "confidence": confidence,
        "source":     "model",
        "shap":       build_shap_explanation(vector),
    }


@app.post("/feedback")
def receive_feedback(data: FeedbackRequest):
    """
    Called when user clicks 'Wrong?' button.
    Flips the predicted label and saves to feedback.csv.
    """
    predicted = data.predicted_label.strip().upper()
    if predicted not in ("FAKE", "REAL"):
        return {"status": "error", "message": "predicted_label must be FAKE or REAL"}

    correct = "REAL" if predicted == "FAKE" else "FAKE"
    _save_feedback(data.text.strip(), predicted, correct)

    return {
        "status":        "saved",
        "correct_label": correct,
        "message":       f"Got it! Marked as {correct} and saved for retraining.",
    }


@app.get("/feedback/stats")
def feedback_stats():
    """Quick overview of collected feedback — useful for admin dashboard."""
    rows = _load_feedback()
    fake_count = sum(1 for r in rows if r.get("correct_label", "").upper() == "FAKE")
    real_count = sum(1 for r in rows if r.get("correct_label", "").upper() == "REAL")
    return {
        "total":          len(rows),
        "corrected_fake": fake_count,
        "corrected_real": real_count,
    }