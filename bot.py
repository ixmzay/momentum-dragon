from flask import Flask
import requests

app = Flask(__name__)

# 🔒 WARNING: Don't use this in production without securing your token
TELEGRAM_TOKEN = "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M"
TELEGRAM_CHAT_ID = "-1002580715831"

def analyze_and_alert():
    message = "✅ Test: ZayMoveBot is alive and hardcoded 🐉"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    print(f"📤 Sending message to {TELEGRAM_CHAT_ID}...")

    response = requests.post(url, json=payload)
    print("📨 Telegram API response:", response.text)

@app.route('/')
def index():
    return "ZayMoveBot (hardcoded) is live."

@app.route('/schedule')
def schedule():
    print("🔥 /schedule route triggered")
    analyze_and_alert()
    return "OK"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
