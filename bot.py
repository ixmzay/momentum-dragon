import requests
import feedparser
from textblob import TextBlob

# === HARDCODED CREDENTIALS ===
TELEGRAM_TOKEN = "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M"
TELEGRAM_CHAT_ID = "-1002580715831"  # your personal chat ID or group ID

RSS_URL = "https://finance.yahoo.com/rss/topstories"

WATCHLIST = {
    "AAPL": ["AAPL", "Apple"],
    "MSFT": ["MSFT", "Microsoft"],
    "GOOGL": ["GOOGL", "Google", "Alphabet"],
    "AMZN": ["AMZN", "Amazon"],
    "META": ["META", "Facebook", "Meta"],
    "TSLA": ["TSLA", "Tesla"],
    "NVDA": ["NVDA", "NVIDIA"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "INTC": ["INTC", "Intel"],
    "NFLX": ["NFLX", "Netflix"],
    "SPY": ["SPY", "S&P 500"],
    "QQQ": ["QQQ", "Nasdaq"],
    "IWM": ["IWM", "Russell 2000"],
    "XOM": ["XOM", "Exxon", "ExxonMobil"],
    "CVX": ["CVX", "Chevron"],
    "OXY": ["OXY", "Occidental"],
    "WMT": ["WMT", "Walmart"],
    "COST": ["COST", "Costco"],
    "TGT": ["TGT", "Target"],
    "HD": ["HD", "Home Depot"],
    "LOW": ["LOW", "Lowe's"],
    "JPM": ["JPM", "JPMorgan"],
    "BAC": ["BAC", "Bank of America"],
    "GS": ["GS", "Goldman Sachs"],
    "MS": ["MS", "Morgan Stanley"],
    "WFC": ["WFC", "Wells Fargo"],
    "BX": ["BX", "Blackstone"],
    "UBER": ["UBER"],
    "LYFT": ["LYFT"],
    "SNOW": ["SNOW", "Snowflake"],
    "PLTR": ["PLTR", "Palantir"],
    "CRM": ["CRM", "Salesforce"],
    "ADBE": ["ADBE", "Adobe"],
    "SHOP": ["SHOP", "Shopify"],
    "PYPL": ["PYPL", "PayPal"],
    "SQ": ["SQ", "Block"],
    "COIN": ["COIN", "Coinbase"],
    "ROKU": ["ROKU"],
    "BABA": ["BABA", "Alibaba"],
    "JD": ["JD", "JD.com"],
    "NIO": ["NIO"],
    "LI": ["LI", "Li Auto"],
    "XPEV": ["XPEV", "XPeng"],
    "LMT": ["LMT", "Lockheed Martin"],
    "NOC": ["NOC", "Northrop Grumman"],
    "RTX": ["RTX", "Raytheon"],
    "BA": ["BA", "Boeing"],
    "GE": ["GE", "General Electric"],
    "CAT": ["CAT", "Caterpillar"],
    "DE": ["DE", "John Deere"],
    "F": ["F", "Ford"],
    "GM": ["GM", "General Motors"],
    "RIVN": ["RIVN", "Rivian"],
    "LCID": ["LCID", "Lucid"],
    "PFE": ["PFE", "Pfizer"],
    "MRNA": ["MRNA", "Moderna"],
    "JNJ": ["JNJ", "Johnson & Johnson"],
    "BMY": ["BMY", "Bristol Myers"],
    "UNH": ["UNH", "UnitedHealth"],
    "MDT": ["MDT", "Medtronic"],
    "ABBV": ["ABBV", "AbbVie"],
    "TMO": ["TMO", "Thermo Fisher"],
    "SHEL": ["SHEL", "Shell"],
    "BP": ["BP", "British Petroleum"],
    "UL": ["UL", "Unilever"],
    "BTI": ["BTI", "British American Tobacco"],
    "SAN": ["SAN", "Santander"],
    "DB": ["DB", "Deutsche Bank"],
    "VTOL": ["VTOL", "Bristow Group"],
    "EVTL": ["EVTL", "Vertical Aerospace"],
    "EH": ["EH", "EHang"],
    "PL": ["PL", "Planet Labs"],
    "TT": ["TT", "Trane"],
    "JCI": ["JCI", "Johnson Controls"],
    "RDW": ["RDW", "Redwire"],
    "LOAR": ["LOAR", "Loar Holdings"],
    "PANW": ["PANW", "Palo Alto Networks"],
    "CRWD": ["CRWD", "CrowdStrike"],
    "NET": ["NET", "Cloudflare"],
    "ZS": ["ZS", "Zscaler"],
    "TSM": ["TSM", "Taiwan Semiconductor"],
    "AVGO": ["AVGO", "Broadcom"],
    "MU": ["MU", "Micron"],
    "TXN": ["TXN", "Texas Instruments"],
    "QCOM": ["QCOM", "Qualcomm"],
}

SENTIMENT_THRESHOLD = 0.1

# Track sent news titles to avoid duplicates during runtime
sent_news = set()

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
        sent_news.add(title)

    print("✅ Done scanning news.\n")

if __name__ == "__main__":
    analyze_news()
