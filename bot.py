import requests
import feedparser
from textblob import TextBlob
from pathlib import Path

# === HARDCODED CREDENTIALS ===
TELEGRAM_TOKEN = "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M"
TELEGRAM_CHAT_ID = "5528794335"
RSS_URL = "https://finance.yahoo.com/rss/topstories"
BENZINGA_API_KEY = "bz.XAO6BCTUMYPFGHXXL7SJ3ZU4IRRTFRE7"
BENZINGA_URL = "https://api.benzinga.com/api/v2/news"

# === CONFIG ===
SENTIMENT_THRESHOLD = 0.1
SENT_LOG_PATH = Path("sent_titles.txt")

# === LOAD SENT NEWS ===
if SENT_LOG_PATH.exists():
    sent_news = set(SENT_LOG_PATH.read_text(encoding="utf-8").splitlines())
else:
    sent_news = set()

# === WATCHLIST ===
WATCHLIST = {
    "AAPL": ["AAPL", "Apple"], "MSFT": ["MSFT", "Microsoft"], "GOOGL": ["GOOGL", "Google", "Alphabet"],
    "AMZN": ["AMZN", "Amazon"], "META": ["META", "Facebook", "Meta"], "TSLA": ["TSLA", "Tesla"],
    "NVDA": ["NVDA", "NVIDIA"], "AMD": ["AMD", "Advanced Micro Devices"], "INTC": ["INTC", "Intel"],
    "NFLX": ["NFLX", "Netflix"], "SPY": ["SPY", "S&P 500"], "QQQ": ["QQQ", "Nasdaq"], "IWM": ["IWM", "Russell 2000"],
    "XOM": ["XOM", "Exxon"], "CVX": ["CVX", "Chevron"], "OXY": ["OXY", "Occidental"], "WMT": ["WMT", "Walmart"],
    "COST": ["COST", "Costco"], "TGT": ["TGT", "Target"], "HD": ["HD", "Home Depot"], "LOW": ["LOW", "Lowe's"],
    "JPM": ["JPM", "JPMorgan"], "BAC": ["BAC", "Bank of America"], "GS": ["GS", "Goldman Sachs"],
    "MS": ["MS", "Morgan Stanley"], "WFC": ["WFC", "Wells Fargo"], "BX": ["BX", "Blackstone"],
    "UBER": ["UBER"], "LYFT": ["LYFT"], "SNOW": ["SNOW"], "PLTR": ["PLTR"], "CRM": ["CRM"],
    "ADBE": ["ADBE"], "SHOP": ["SHOP"], "PYPL": ["PYPL"], "SQ": ["SQ"], "COIN": ["COIN"], "ROKU": ["ROKU"],
    "BABA": ["BABA"], "JD": ["JD"], "NIO": ["NIO"], "LI": ["LI"], "XPEV": ["XPEV"], "LMT": ["LMT"],
    "NOC": ["NOC"], "RTX": ["RTX"], "BA": ["BA"], "GE": ["GE"], "CAT": ["CAT"], "DE": ["DE"],
    "F": ["F"], "GM": ["GM"], "RIVN": ["RIVN"], "LCID": ["LCID"], "PFE": ["PFE"], "MRNA": ["MRNA"],
    "JNJ": ["JNJ"], "BMY": ["BMY"], "UNH": ["UNH"], "MDT": ["MDT"], "ABBV": ["ABBV"], "TMO": ["TMO"],
    "SHEL": ["SHEL"], "BP": ["BP"], "UL": ["UL"], "BTI": ["BTI"], "SAN": ["SAN"], "DB": ["DB"],
    "VTOL": ["VTOL"], "EVTL": ["EVTL"], "EH": ["EH"], "PL": ["PL"], "TT": ["TT"], "JCI": ["JCI"],
    "RDW": ["RDW"], "LOAR": ["LOAR"], "PANW": ["PANW"], "CRWD": ["CRWD"], "NET": ["NET"], "ZS": ["ZS"],
    "TSM": ["TSM"], "AVGO": ["AVGO"], "MU": ["MU"], "TXN": ["TXN"], "QCOM": ["QCOM"]
}

# === FUNCTIONS ===

def calculate_confidence(headline):
    headline_lower = headline.lower()
    strong_words = [
        "upgrade", "upgraded", "downgrade", "downgraded", "raise", "raised", "cut", "cuts",
        "record", "beat", "beats", "miss", "misses", "warn", "warning", "surge", "surges",
        "fall", "falls", "breakout", "breakouts", "outperform", "outperforming",
        "accelerate", "accelerated", "accelerating", "beat consensus", "cut guidance",
        "miss estimates", "beat estimates"
    ]
    moderate_words = [
        "gain", "gains", "growth", "growing", "buy", "buying", "strong", "positive", "profit", "profits",
        "expansion", "increase", "increased", "increases", "decline", "declines", "weak", "weakness",
        "loss", "losses", "drop", "drops", "lower", "lowered", "lowering", "sell", "selling",
        "rebound", "rebounded", "margin expansion", "pullback", "pulled back"
    ]
    strong_hits = sum(phrase in headline_lower for phrase in strong_words)
    moderate_hits = sum(phrase in headline_lower for phrase in moderate_words)
    score = min(100, strong_hits * 25 + moderate_hits * 10)

    if score >= 70:
        confidence_label = "High"
    elif score >= 30:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"
    return score, confidence_label

def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
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
        title = entry.title.strip()
        if title in sent_news:
            print(f"⚠️ Duplicate news skipped: {title}")
            continue

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

        confidence_score, confidence_label = calculate_confidence(title)
        if confidence_label == "Low":
            print("⚠️ Low confidence — skipping alert")
            continue

        message = (
            f"*{direction} News on {ticker}:*\n"
            f"{title}\n\n"
            f"_Confidence:_ {confidence_score}% ({confidence_label})"
        )

        send_to_telegram(message)
        title = title.strip()
        if title not in sent_news:
            sent_news.add(title)
            with SENT_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(f"{title}\n")

    print("✅ Done scanning Yahoo RSS news.\n")

def fetch_benzinga_news():
    headers = {"Authorization": f"Bearer {BENZINGA_API_KEY}"}
    params = {"items": 50, "tickers": ",".join(WATCHLIST.keys())}
    try:
        resp = requests.get(BENZINGA_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("news", [])
    except Exception as e:
        print(f"❌ Benzinga API error: {e}")
        return []

def analyze_benzinga_news():
    print("🚨 Running analyze_benzinga_news()")
    news_list = fetch_benzinga_news()

    for article in news_list:
        title = article.get("title", "").strip()
        if not title or title in sent_news:
            print(f"⚠️ Duplicate or empty Benzinga news skipped: {title}")
            continue

        ticker = match_watchlist(title)
        if not ticker:
            continue

        sentiment = analyze_sentiment(title)
        print(f"📊 Benzinga Sentiment Score: {sentiment:.2f}")
        if sentiment > SENTIMENT_THRESHOLD:
            direction = "Bullish"
        elif sentiment < -SENTIMENT_THRESHOLD:
            direction = "Bearish"
        else:
            print("⚠️ Neutral Benzinga sentiment — skipping alert")
            continue

        confidence_score, confidence_label = calculate_confidence(title)
        if confidence_label == "Low":
            print("⚠️ Low Benzinga confidence — skipping alert")
            continue

        message = (
            f"*{direction} News on {ticker}:*\n"
            f"{title}\n\n"
            f"_Confidence:_ {confidence_score}% ({confidence_label})"
        )

        send_to_telegram(message)
        title = title.strip()
        if title not in sent_news:
            sent_news.add(title)
            with SENT_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(f"{title}\n")

    print("✅ Done scanning Benzinga news.\n")

# === RUN ===
if __name__ == "__main__":
    analyze_news()
    analyze_benzinga_news()
