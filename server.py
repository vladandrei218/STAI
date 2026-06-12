from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
import re

app = Flask(__name__)
CORS(app)

# Load model + vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("fakenews_model.pkl")

def extract_domain(url):
    if not url:
        return ""
    match = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    return match.group(1).lower() if match else ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

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

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    title = data.get("title", "")
    url   = data.get("url", "")

    domain   = extract_domain(url)
    cleaned  = clean_text(title) + " " + domain

    tfidf_vec = tfidf.transform([cleaned])
    hc_vec    = csr_matrix(handcrafted_features([cleaned]))
    features  = hstack([tfidf_vec, hc_vec])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]  # [prob_real, prob_fake]
    confidence = round(float(max(proba)) * 100, 1)

    return jsonify({
        "label":      "FAKE" if prediction == 1 else "REAL",
        "confidence": confidence,
        "prob_real":  round(float(proba[0]) * 100, 1),
        "prob_fake":  round(float(proba[1]) * 100, 1),
        "title":      title
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)