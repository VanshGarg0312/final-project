import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FEEDBACK_FILE   = "feedback.csv"
MODEL_PATH      = "saved_models/model.pkl"
VECTORIZER_PATH = "saved_models/vectorizer.pkl"
BACKGROUND_PATH = "saved_models/background.pkl"
MAX_FEATURES    = 5000
TEST_SIZE       = 0.2
FEEDBACK_WEIGHT = 5          # repeat each feedback row N times (priority boost)

os.makedirs("saved_models", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD BASE DATASET
# ─────────────────────────────────────────────
print("📂 Loading base dataset...")
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# 0 = FAKE, 1 = REAL  ← explicit and documented
fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)

# ── Sanity check ──────────────────────────────
print("\n✅ Label distribution in base dataset:")
print(data["label"].value_counts().rename({0: "FAKE (0)", 1: "REAL (1)"}))

# Keep only rows where 'text' column is non-empty
if "text" not in data.columns:
    raise ValueError("❌ Dataset must have a 'text' column. Found: " + str(data.columns.tolist()))

data = data[["text", "label"]].dropna(subset=["text"])
data["text"] = data["text"].astype(str).str.strip()
data = data[data["text"] != ""]

# ─────────────────────────────────────────────
# 2. LOAD & MERGE FEEDBACK (with priority boost)
# ─────────────────────────────────────────────
if os.path.exists(FEEDBACK_FILE):
    print(f"\n📋 Loading feedback from {FEEDBACK_FILE}...")
    feedback = pd.read_csv(FEEDBACK_FILE)

    required_cols = {"text", "correct_label"}
    if required_cols.issubset(feedback.columns):
        feedback = feedback[["text", "correct_label"]].dropna()
        feedback["text"] = feedback["text"].astype(str).str.strip()
        feedback = feedback[feedback["text"] != ""]

        # Map string labels → int
        label_map = {"FAKE": 0, "REAL": 1}
        feedback["label"] = feedback["correct_label"].str.upper().map(label_map)
        feedback = feedback.dropna(subset=["label"])
        feedback["label"] = feedback["label"].astype(int)
        feedback = feedback[["text", "label"]]

        print(f"   Found {len(feedback)} feedback rows.")
        print("   Feedback label distribution:")
        print(feedback["label"].value_counts().rename({0: "FAKE (0)", 1: "REAL (1)"}))

        # Repeat feedback rows to give them priority weight
        feedback_boosted = pd.concat([feedback] * FEEDBACK_WEIGHT, ignore_index=True)
        data = pd.concat([data, feedback_boosted], ignore_index=True)
        print(f"   ✅ Merged feedback (each row counted {FEEDBACK_WEIGHT}x for priority).")
    else:
        print(f"   ⚠️  Skipping feedback — missing columns. Expected: {required_cols}, got: {set(feedback.columns)}")
else:
    print(f"\n⚠️  No feedback file found at '{FEEDBACK_FILE}'. Training on base dataset only.")

# ─────────────────────────────────────────────
# 3. SHUFFLE
# ─────────────────────────────────────────────
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data["text"]
y = data["label"]

print(f"\n📊 Final dataset size: {len(data)} rows")
print("   Final label distribution:")
print(y.value_counts().rename({0: "FAKE (0)", 1: "REAL (1)"}))

# ─────────────────────────────────────────────
# 4. VECTORIZE
# ─────────────────────────────────────────────
print("\n🔠 Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    stop_words="english",
    ngram_range=(1, 2),      # unigrams + bigrams for better context
    sublinear_tf=True,       # dampens very frequent terms
)
X_vec = vectorizer.fit_transform(X)

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 6. TRAIN MODEL
# ─────────────────────────────────────────────
print("\n🤖 Training Logistic Regression...")
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",   # handles any residual imbalance
    solver="lbfgs",
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────────
pred = model.predict(X_test)

print("\n📈 Evaluation Results:")
print(f"   Accuracy : {accuracy_score(y_test, pred):.4f}")
print("\n   Classification Report:")
print(classification_report(y_test, pred, target_names=["FAKE (0)", "REAL (1)"]))

print("   Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, pred)
print(f"              Pred FAKE  Pred REAL")
print(f"   Actual FAKE   {cm[0][0]:<8}  {cm[0][1]}")
print(f"   Actual REAL   {cm[1][0]:<8}  {cm[1][1]}")

# ── Warn if model still biased ────────────────
fake_as_real = cm[0][1]
real_as_fake = cm[1][0]
if real_as_fake > fake_as_real * 1.5:
    print("\n⚠️  WARNING: Model is still predicting REAL news as FAKE more than expected.")
    print("   → Check label mapping in your dataset or add more feedback corrections.")

# ─────────────────────────────────────────────
# 8. VERIFY LABEL DIRECTION (quick sanity test)
# ─────────────────────────────────────────────
print("\n🔍 Label direction sanity check:")
print(f"   model.classes_ = {model.classes_}")
print(f"   → Index 0 = class {model.classes_[0]} ({'FAKE' if model.classes_[0] == 0 else 'REAL'})")
print(f"   → Index 1 = class {model.classes_[1]} ({'REAL' if model.classes_[1] == 1 else 'FAKE'})")
print("   (predict_proba columns follow this order)")

# ─────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────
joblib.dump(model,      MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(X_train[:100], BACKGROUND_PATH)

print(f"\n✅ Model saved      → {MODEL_PATH}")
print(f"✅ Vectorizer saved → {VECTORIZER_PATH}")
print(f"✅ Background saved → {BACKGROUND_PATH}")
print("\n🎉 Training complete!")