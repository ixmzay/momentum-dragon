import os
from flask import Flask

app = Flask(__name__)

@app.route("/schedule")
def schedule():
    print("🔥 /schedule HIT")
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
