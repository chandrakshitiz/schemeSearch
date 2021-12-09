from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods = ['POST'])
def submit():
    if request.method == "POST":
        name = request.form["search"]
    return render_template("search.html", n = model.recommendations(name), t = name)

if __name__ == "__main__":
    app.run(debug=True)
