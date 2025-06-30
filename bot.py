import requests
import feedparser
from textblob import TextBlob

# === HARDCODED CREDENTIALS ===
TELEGRAM_TOKEN = "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M"
TELEGRAM_CHAT_ID = "5528794335"  # your personal chat ID or group ID

RSS_URL = "https://finance.yahoo.com/rss/topstories"

WATCHLIST = {
    "TSLA": ["TSLA", "Tesla"],
    "AAPL": ["AAPL", "Apple"],
    "NVDA": ["NVDA", "NVIDIA"],
    "QQQ": ["QQQ"],
    "SPY": ["SPY"],
    "CRCL": ["CRCL", "Circle", "Circle Internet"],
    "PLTR": ["PLTR", "Palantir"],
    "COIN": ["COIN", "Coinbase"],
    "HOOD": ["HOOD", "Robinhood"],
    "CIRCL": ["CIRCL", "Circle"],
    "WMT": ["WMT", "Walmart"],
    "COST": ["COST", "Costco"]
}

SENTIMENT_THRESHOLD = 0.1

def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        r = requests.post(url, json=payload)
        print(f"✅ Telegram sent! Status: {r.status_code}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def match_watchlist(text):
    text_lower = text.lower()
    for ticker, keywords in WATCHLIST.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                print(f"✅ Matched {ticker} via keyword '{keyword}' in: {text}")
                return ticker
    print(f"❌ No match found in: {text}")
    return None

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def analyze_news():
    print("🚨 Running analyze_news()")
    feed = feedparser.parse(RSS_URL)
    print(f"📥 Pulled {len(feed.entries)} RSS entries")

    for entry in feed.entries:
        title = entry.title
        print(f"🔍 Checking title: {title}")

        ticker = match_watchlist(title)
        if not ticker:
            continue

        sentiment = analyze_sentiment(title)
        print(f"📊 Sentiment Score: {sentiment:.2f}")

        if sentiment > SENTIMENT_THRESHOLD:
            direction = "Bullish"
        elif sentiment < -SENTIMENT_THRESHOLD:
            direction = "Bearish"
        else:
            print("⚠️ Neutral sentiment — skipping alert")
            continue

        message = f"*{direction} News on {ticker}:*\n{title}"
        send_to_telegram(message)

    print("✅ Done scanning news.\n")

if __name__ == "__main__":
    analyze_news()
