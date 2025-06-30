from flask import Flask
import requests
import feedparser
from datetime import datetime
import pytz

app = Flask(__name__)

# === CONFIG ===
TELEGRAM_TOKEN = "<7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M>"
TELEGRAM_CHAT_ID = "<-1002580715831>"
RSS_FEED_URL = "https://finance.yahoo.com/rss/topstories"
WATCHLIST = ["TSLA", "QQQ", "SPY", "NVDA", "AAPL"]
KEYWORDS_BULLISH = ["beats", "surges", "record", "deal", "acquires", "raises", "rips", "expands", "invests"]
CONFIDENCE_THRESHOLD = 50

# === TELEGRAM SEND ===
def send_to_telegram(message):
    print("📨 Telegram Payload:", message)
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    response = requests.post(url, json=payload)
    print("✅ Telegram Response:", response.json())

# === WATCHLIST MATCH ===
def match_watchlist(text):
    for ticker in WATCHLIST:
        if ticker.lower() in text.lower():
            return ticker
    return None

# === SENTIMENT SCORE ===
def score_sentiment(text):
    score = 0
    for keyword in KEYWORDS_BULLISH:
        if keyword.lower() in text.lower():
            score += 15
    return min(score, 100)

# === MAIN LOGIC ===
def analyze_news():
    print("🚨 analyze_news() IS RUNNING")
    feed = feedparser.parse(RSS_FEED_URL)
    print(f"✅ Pulled {len(feed.entries)} entries from RSS")

    for entry in feed.entries:
        title = entry.title
        link = entry.link
        print(f"🔍 Checking title: {title}")

        matched_ticker = match_watchlist(title)
        if matched_ticker:
            print(f"🎯 Matched ticker: {matched_ticker}")
            score = score_sentiment(title)
            print(f"📊 Sentiment score: {score}")

            if score >= CONFIDENCE_THRESHOLD:
                message = f"\ud83d\udcca News Momentum Alert - ${matched_ticker}\n"
                message += f"Sentiment: {'Bullish' if score > 55 else 'Bearish'} | Confidence: {score}%\n"
                message += f"Headline: {title}\n"
                message += f"\ud83d\udd17 {link}"
                print("📤 Sending alert to Telegram...")
                send_to_telegram(message)
                break
        else:
            print("❌ No ticker match")

# === ROUTES ===
@app.route('/')
def home():
    return "ZayMoveBot v2 is alive!"

@app.route('/schedule')
def schedule():
    print("🔥 SCHEDULE ENDPOINT HIT — ANALYZE_NEWS STARTING")
    analyze_news()
    print("✅ analyze_news() finished running")
    return "OK"

# === MAIN ===
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
