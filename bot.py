import os, requests, time
from flask import Flask
from bs4 import BeautifulSoup
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
app = Flask(__name__)

def send_telegram(msg):
    print("Preparing to send Telegram message...")  # Force it to show in logs
    try:
        token = os.getenv("TELEGRAM_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        print("Token loaded:", bool(token))
        print("Chat ID loaded:", bool(chat_id))
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        res = requests.post(url, data={"chat_id": chat_id, "text": msg})
        print("Message:", msg)
        print("Status:", res.status_code)
        print("Response:", res.text)
    except Exception as e:
        print("Error sending Telegram message:", e)



def fetch_yahoo_headlines():
    resp = requests.get("https://finance.yahoo.com/")
    soup = BeautifulSoup(resp.text, "html.parser")
    return [a.get_text() for a in soup.select("h3 a[href^='/news/']")[:5]]

def analyze_and_alert():
    for h in fetch_yahoo_headlines():
        polarity = TextBlob(h).sentiment.polarity
        sentiment = "Bullish" if polarity > 0.1 else "Bearish" if polarity < -0.1 else "Neutral"
        msg = f"🚨 [NEWS ALERT]\n📰 {h}\n🧠 Sentiment: {sentiment}\n🕒 {time.strftime('%Y‑%m‑%d %H:%M:%S')}"
        send_telegram(msg)
        time.sleep(1)

@app.route("/schedule")
def schedule():
    analyze_and_alert()
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
