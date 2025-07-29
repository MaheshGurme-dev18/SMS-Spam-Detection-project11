from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        message = request.form["message"]
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        return render_template("index.html", prediction=prediction, message=message)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)