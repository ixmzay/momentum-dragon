import os
import time
import requests
from flask import Flask
from bs4 import BeautifulSoup
from textblob import TextBlob
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = Flask(__name__)

def send_telegram(msg):
    print("🚀 Preparing to send Telegram message...")
    try:
        print("✅ Token loaded:", bool(TOKEN))
        print("✅ Chat ID loaded:", bool(CHAT_ID))
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        res = requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
        print("📨 Message sent:", msg)
        print("📡 Status:", res.status_code)
        print("📬 Telegram response:", res.text)
    except Exception as e:
        print("❌ Error sending message:", e)

def fetch_yahoo_headlines():
    print("🔎 Fetching Yahoo Finance headlines...")
    try:
        resp = requests.get("https://finance.yahoo.com/")
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = [a.get_text() for a in soup.select("h3 a[href^='/news/']")[:5]]
        print(f"📰 Found {len(headlines)} headlines")
        return headlines
    except Exception as e:
        print("❌ Error scraping headlines:", e)
        return []

def analyze_and_alert():
    print("🧠 Running analyze_and_alert...")
    headlines = fetch_yahoo_headlines()
    for h in headlines:
        polarity = TextBlob(h).sentiment.polarity
        sentiment = "Bullish" if polarity > 0.1 else "Bearish" if polarity < -0.1 else "Neutral"
        msg = f"🚨 [NEWS ALERT]\n📰 {h}\n🧠 Sentiment: {sentiment}\n🕒 {time.strftime('%Y-%m-%d %H:%M:%S')}"
        print("📊 Headline:", h, "| Sentiment:", sentiment)
        send_telegram(msg)
        time.sleep(1)

@app.route("/schedule")
def schedule():
    print("⚡️ /schedule route was hit")
    analyze_and_alert()
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
