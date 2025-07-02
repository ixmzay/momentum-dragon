import time
import re
import json
import threading
from pathlib import Path

import requests
import feedparser
import yfinance as yf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M"
TELEGRAM_CHAT_ID = "5528794335"

# === RSS & BENZINGA CONFIG ===
RSS_URL           = "https://finance.yahoo.com/rss/topstories"
BENZINGA_API_KEY  = "bz.XAO6BCTUMYPFGHXXL7SJ3ZU4IRRTFRE7"
BENZINGA_URL      = "https://api.benzinga.com/api/v2/news"

# === THRESHOLDS & LIMITS ===
CONFIDENCE_THRESHOLD  = 30     # Keyword-score threshold
RATE_LIMIT_SECONDS    = 1800   # 30-minute cooldown

# === LOG FILE PATHS ===
SENT_LOG_PATH       = Path("sent_titles.txt")
RATE_LIMIT_LOG_PATH = Path("rate_limit.json")
TRAINING_DATA_PATH  = Path("training_data.json")

# === WATCHLIST ===
WATCHLIST = {
    "AAPL": ["AAPL", "Apple"],
    # … include all your tickers here …
    "QCOM": ["QCOM", "Qualcomm"]
}

# === Persistent storage ===
sent_news       = set(SENT_LOG_PATH.read_text(encoding="utf-8").splitlines()) if SENT_LOG_PATH.exists() else set()
rate_limit_data = json.loads(RATE_LIMIT_LOG_PATH.read_text(encoding="utf-8")) if RATE_LIMIT_LOG_PATH.exists() else {}
training_data   = json.loads(TRAINING_DATA_PATH.read_text(encoding="utf-8")) if TRAINING_DATA_PATH.exists() else {"texts": [], "labels": []}

# === ML Model ===
vectorizer  = TfidfVectorizer(stop_words="english", max_features=500)
model       = LogisticRegression()
model_lock  = threading.Lock()

def train_model():
    texts  = training_data.get("texts", [])
    labels = training_data.get("labels", [])
    if len(texts) < 10:
        print("⚠️ Not enough training data to train ML model.")
        return
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

def classify_text(text: str):
    if not training_data.get("texts"):
        return None, None
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max() * 100
    return pred, round(prob, 2)

# === FinBERT Setup ===
finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model     = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# === Confidence Keywords ===
PRIORITY_KEYWORDS = [
    "earnings", "upgrade", "downgrade", "price target", "beat estimates",
    "miss estimates", "warning", "lawsuit", "guidance", "dividend",
    "buyback", "merger", "acquisition", "ipo", "layoff",
    "revenue", "profit"
]

def calculate_confidence(headline: str) -> (int, str):
    hl = headline.lower()
    strong = ["upgrade", "beat estimates", "record", "surge", "outperform", "raise", "warning", "cut"]
    moderate = ["buy", "positive", "growth", "profit", "decline", "drop", "loss"]
    priority = PRIORITY_KEYWORDS

    s_hits = sum(w in hl for w in strong)
    m_hits = sum(w in hl for w in moderate)
    p_hits = sum(w in hl for w in priority)
    score = min(100, s_hits * 25 + m_hits * 10 + p_hits * 15)
    label = "High" if score >= 70 else "Medium" if score >= CONFIDENCE_THRESHOLD else "Low"
    return score, label

def get_vix_level():
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d", interval="1m")
        if hist.empty:
            return None, "⚠️ VIX data unavailable"

        latest_vix = hist["Close"].iloc[-1]
        if latest_vix < 14:
            label = "🟢 Low Fear (Bullish Conditions)"
        elif latest_vix < 20:
            label = "🟡 Normal (Watch Key Levels)"
        elif latest_vix < 25:
            label = "🟠 Caution Zone (Hedge Suggested)"
        elif latest_vix < 30:
            label = "🔴 High Fear (Volatile & Risky)"
        else:
            label = "🚨 Panic Mode (Avoid Large Exposure)"
        return round(latest_vix, 2), label
    except Exception as e:
        return None, f"❌ VIX error: {e}"

def send_to_telegram(message: str):
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=5)
        print("Telegram response:", resp.status_code, resp.text)
        resp.raise_for_status()
        print("✅ Telegram alert sent.")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def is_rate_limited(ticker: str) -> bool:
    last = rate_limit_data.get(ticker, 0)
    if time.time() - last < RATE_LIMIT_SECONDS:
        print(f"⏳ Rate limited: {ticker}")
        return True
    return False

def update_rate_limit(ticker: str):
    rate_limit_data[ticker] = time.time()
    RATE_LIMIT_LOG_PATH.write_text(json.dumps(rate_limit_data), encoding="utf-8")

def match_watchlist(text: str) -> str | None:
    tl = text.lower()
    for ticker, kws in WATCHLIST.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw.lower())}\b", tl):
                return ticker
    return None

def analyze_sentiment(text: str) -> float:
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = finbert_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).detach().numpy()[0]
    return probs[2] - probs[0]  # pos - neg

def should_send_alert(title, ticker, conf_score=None):
    if title in sent_news or is_rate_limited(ticker):
        return False
    return True

def send_alert(title, ticker, sentiment, conf_score, conf_label, source):
    vix_val, vix_lbl = get_vix_level()
    ml_pred, ml_conf = classify_text(title)
    msg = (
        f"🗞 *{source} Alert*\n"
        f"*{ticker}* — {title}\n"
        f"📊 Sentiment: `{sentiment:.2f}`\n"
        f"🎯 Confidence: *{conf_score}%* ({conf_label})"
    )
    if ml_pred:
        msg += f"\n🤖 ML: *{ml_pred}* ({ml_conf}%)"
    msg += (
        f"\n🌪 VIX: *{vix_val}* — {vix_lbl}\n"
        f"🕒 {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    send_to_telegram(msg)
    sent_news.add(title)
    SENT_LOG_PATH.write_text("\n".join(sent_news), encoding="utf-8")
    update_rate_limit(ticker)

# === Yahoo RSS ===
def process_yahoo_entry(entry):
    title = entry.get("title", "").strip()
    print("▶️ Yahoo headline:", title)
    ticker = match_watchlist(title)
    print("   → matched ticker:", ticker)
    if not ticker: return
    sentiment = analyze_sentiment(title)
    conf_score, conf_label = calculate_confidence(title)
    if should_send_alert(title, ticker):
        send_alert(title, ticker, sentiment, conf_score, conf_label, "Yahoo")

def analyze_yahoo():
    print("📡 Scanning Yahoo RSS...")
    feed = feedparser.parse(RSS_URL)
    for entry in feed.entries:
        process_yahoo_entry(entry)
    print("✅ Yahoo done.")

# === Benzinga ===
def fetch_benzinga(chunk):
    try:
        resp = requests.get(BENZINGA_URL, params={
            "tickers": ",".join(chunk),
            "items": 50,
            "token": BENZINGA_API_KEY
        }, timeout=10)
        resp.raise_for_status()
        return resp.json().get("news", [])
    except Exception as e:
        print(f"❌ Benzinga error: {e}")
        return []

def process_benzinga_article(article):
    title = article.get("title", "").strip()
    print("▶️ Benzinga headline:", title)
    ticker = match_watchlist(title)
    print("   → matched ticker:", ticker)
    if not ticker: return
    sentiment = analyze_sentiment(title)
    conf_score, conf_label = calculate_confidence(title)
    if should_send_alert(title, ticker):
        send_alert(title, ticker, sentiment, conf_score, conf_label, "Benzinga")

def analyze_benzinga():
    print("📡 Scanning Benzinga...")
    tickers = list(WATCHLIST.keys())
    for i in range(0, len(tickers), 20):
        for art in fetch_benzinga(tickers[i:i+20]):
            process_benzinga_article(art)
    print("✅ Benzinga done.")

def main():
    print("🚀 Starting market bot...")
    train_model()
    while True:
        try:
            analyze_yahoo()
            analyze_benzinga()
            print("⏲ Sleeping 60 sec...\n")
            time.sleep(60)
        except Exception as e:
            print(f"💥 Main loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
