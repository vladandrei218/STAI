import pandas as pd

# Load all 4 files
gos_real = pd.read_csv("gossipcop_real.csv")
gos_fake = pd.read_csv("gossipcop_fake.csv")
pol_real = pd.read_csv("politifact_real.csv")
pol_fake = pd.read_csv("politifact_fake.csv")

# Add labels
gos_real["label"] = 0  # real
gos_fake["label"] = 1  # fake
pol_real["label"] = 0
pol_fake["label"] = 1

# Add source column (optional but useful)
gos_real["source"] = "gossipcop"
gos_fake["source"] = "gossipcop"
pol_real["source"] = "politifact"
pol_fake["source"] = "politifact"

# Combine everything
df = pd.concat([gos_real, gos_fake, pol_real, pol_fake], ignore_index=True)

print(df.shape)
print(df.columns.tolist())  # check what columns you have
print(df.head(2))


# STEP 2 

import re

def extract_domain(url):
    if pd.isna(url):
        return ""
    match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    return match.group(1).lower() if match else ""

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_title"] = df["title"].apply(clean_text)
df["domain"] = df["news_url"].apply(extract_domain)

# Combine title + domain as input text
df["clean_text"] = df["clean_title"] + " " + df["domain"]

# Drop empty
df = df[df["clean_title"].str.len() > 0].reset_index(drop=True)

print(f"Dataset size: {len(df)}")
print(df["label"].value_counts())



# STEP 3
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Handcrafted features
def handcrafted_features(texts):
    sensational = [
        # Original
        "shocking", "unbelievable", "secret", "exposed", "truth",
        "hoax", "breaking", "urgent", "warning",
        # Conspiracy / health misinfo
        "conspiracy", "coverup", "hidden", "miracle", "cure",
        "bigpharma", "alien", "government", "reveal", "trick",
        "weird", "mainstream", "wake", "sheeple", "banned",
        "censored", "suppressed", "deepstate", "fake",
        # Political misinfo
        "rigged", "stolen", "corrupt", "traitor", "criminal",
        "arrest", "indicted", "coup", "invasion", "collapse",
        # Emotional bait
        "terrifying", "disgusting", "outrageous", "heartbreaking",
        "destroyed", "obliterated", "slammed", "blasted", "rips"
    ]

    clickbait_patterns = [
        r"you won.t believe",
        r"doctors? (hate|reveal|discovered|don.t want)",
        r"one weird trick",
        r"\d+ (things|reasons|ways|secrets|facts|signs)",
        r"what happened next",
        r"they don.t want you to know",
        r"big.?pharma",
        r"the truth about",
        r"wake up",
        r"share before (this is )?deleted",
        r"mainstream media (won.t|ignores|hides)",
        r"what (they|the media) (aren.t|won.t|don.t)",
        r"this will (shock|change|blow)",
        r"nobody is talking about",
        r"(just|breaking):?\s",
        r"must (see|read|watch|share)",
        r"goes viral",
        r"destroys? (with|in|on)",
        r"can.t believe",
        r"here.s what really",
        r"the real reason",
        r"(secret|hidden) (agenda|truth|plan|cure)",
    ]

    feats = []
    for text in texts:
        words = text.split()

        # Basic features
        exclamation  = text.count("!") / (len(text) + 1)
        question     = text.count("?") / (len(text) + 1)
        caps_ratio   = sum(1 for c in text if c.isupper()) / (len(text) + 1)
        word_count   = len(words)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        # Sensational word count
        sens_count = sum(1 for w in words if w in sensational)

        # Clickbait pattern count
        clickbait_score = sum(
            1 for p in clickbait_patterns if re.search(p, text.lower())
        )

        # Extra signals
        has_number    = int(bool(re.search(r"\d+", text)))         # "5 secrets"
        all_caps_word = sum(1 for w in words if w.isupper() and len(w) > 2)  # BREAKING
        hedge_words   = ["allegedly", "reportedly", "claims", "sources say",
                         "according to", "officials say", "experts say"]
        hedge_count   = sum(1 for h in hedge_words if h in text.lower())

        feats.append([
            exclamation, question, caps_ratio,
            word_count, avg_word_len,
            sens_count, clickbait_score,
            has_number, all_caps_word, hedge_count
        ])

    return np.array(feats)

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")


# STEP 4

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.sparse import hstack
from sklearn.preprocessing import FunctionTransformer

# TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Handcrafted
X_train_hc = handcrafted_features(X_train.tolist())
X_test_hc = handcrafted_features(X_test.tolist())

from scipy.sparse import csr_matrix
X_train_final = hstack([X_train_tfidf, csr_matrix(X_train_hc)])
X_test_final = hstack([X_test_tfidf, csr_matrix(X_test_hc)])

# Train multiple models and compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "LinearSVC":           LinearSVC(max_iter=2000, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_final, y_train)
    print(f"Trained: {name}")


# STEP 5

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

for name, model in models.items():
    preds = model.predict(X_test_final)
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(classification_report(y_test, preds, target_names=["Real", "Fake"]))

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"cm_{name.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()


# STEP 6

from sklearn.model_selection import StratifiedKFold, cross_val_score

best_model = LinearSVC(max_iter=2000, C=1.0)  # swap if LR wins above

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_model, X_train_final, y_train,
                         cv=cv, scoring="f1_macro")

print(f"\n5-Fold CV F1 (macro): {scores.mean():.4f} ± {scores.std():.4f}")


# STEP 7
import joblib

# Save the vectorizer and model separately
from sklearn.calibration import CalibratedClassifierCV

svc = LinearSVC(max_iter=2000, C=1.0)
calibrated_model = CalibratedClassifierCV(svc, cv=5)
calibrated_model.fit(X_train_final, y_train)

joblib.dump(calibrated_model, "fakenews_model.pkl")

print("Model saved!")