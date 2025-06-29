from flask import Flask
import requests
from datetime import datetime
import pytz
import feedparser

app = Flask(__name__)

# === CONFIG ===
TELEGRAM_TOKEN = "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M"
TELEGRAM_CHAT_ID = "-1002580715831"
WATCHLIST = ["TSLA", "QQQ", "SPY", "NVDA", "AAPL"]  # Update as needed
CONFIDENCE_THRESHOLD = 70  # Minimum score to alert

KEYWORDS_BULLISH = ["beats", "surges", "record", "deal", "acquires", "raises", "rips", "expands", "invests"]
KEYWORDS_BEARISH = ["misses", "falls", "lawsuit", "downgrade", "cuts", "drops", "recall", "loss"]

RSS_FEED_URL = "https://finance.yahoo.com/rss/topstories"

# === HELPERS ===
def is_market_open():
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    return now.weekday() < 5 and (
        (now.hour == 9 and now.minute >= 25) or
        (10 <= now.hour < 16) or
        (now.hour == 16 and now.minute == 0)
    )

def score_sentiment(title):
    score = 0
    title_lower = title.lower()
    for word in KEYWORDS_BULLISH:
        if word in title_lower:
            score += 15
    for word in KEYWORDS_BEARISH:
        if word in title_lower:
            score -= 15
    return max(0, min(100, 50 + score))

def match_watchlist(title):
    for symbol in WATCHLIST:
        if symbol in title.upper():
            return symbol
    return None

def analyze_news():
    feed = feedparser.parse(RSS_FEED_URL)
    for entry in feed.entries:
        title = entry.title
        link = entry.link
        matched_ticker = match_watchlist(title)
        if matched_ticker:
            score = score_sentiment(title)
            if score >= CONFIDENCE_THRESHOLD:
                message = f"\ud83d\udcca News Momentum Alert - ${matched_ticker}\n"
                message += f"Sentiment: {'Bullish' if score > 55 else 'Bearish'} | Confidence: {score}%\n"
                message += f"Headline: {title}\n"
                message += f"\ud83d\udd17 {link}"
                send_to_telegram(message)
                break  # Prevent spamming multiple alerts per cycle

def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

@app.route('/')
def index():
    return "ZayMoveBot v2 is live"

@app.route('/schedule')
def schedule():
    if is_market_open():
        analyze_news()
    return "OK"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
