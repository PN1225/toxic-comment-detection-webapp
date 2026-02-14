from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model & vectorizer
with open("toxic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    comment = ""

    if request.method == "POST":
        comment = request.form["comment"]

        # Vectorize comment
        comment_vec = vectorizer.transform([comment])

        # Predict
        prediction = model.predict(comment_vec)[0]

        if prediction == 1:
            result = "⚠️ Toxic comment detected. Comment hidden for safety."
            comment = ""
        else:
            result = "✅ Comment is SAFE"

    return render_template("index.html", result=result, comment=comment)

if __name__ == "__main__":
    app.run(debug=True)
