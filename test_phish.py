from flask import Flask, render_template, request
import joblib
import re
import string

model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)  # ← you were missing this

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = ""
    if request.method == "POST":
        message = request.form["message"]
        cleaned = preprocess_text(message)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)[0]
        prediction = "🛑 Phishing Message" if result == 1 else "✅ Legitimate Message"
    return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))