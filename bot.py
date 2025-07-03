import time
import re
import json
import threading
from pathlib import Path
from datetime import datetime

import requests
import feedparser
import yfinance as yf

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# FinBERT imports
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# Synonym expansion
import nltk
from nltk.corpus import wordnet

# Ensure WordNet is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M"
TELEGRAM_CHAT_ID = "5528794335"

# === RSS & BENZINGA CONFIG ===
RSS_URL           = "https://finance.yahoo.com/rss/topstories"
BENZINGA_API_KEY  = "bz.XAO6BCTUMYPFGHXXL7SJ3ZU4IRRTFRE7"
BENZINGA_URL      = "https://api.benzinga.com/api/v2/news"

# === THRESHOLDS & LIMITS ===
CONFIDENCE_THRESHOLD  = 60     # Medium cutoff raised to 60
RATE_LIMIT_SECONDS    = 1800   # 30-minute cooldown per ticker

# === LOG FILE PATHS ===
SENT_LOG_PATH       = Path("sent_titles.txt")
RATE_LIMIT_LOG_PATH = Path("rate_limit.json")
TRAINING_DATA_PATH  = Path("training_data.json")

# === WATCHLIST ===
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
    "QCOM": ["QCOM", "Qualcomm"]
}

# === BASE KEYWORDS ===
BASE_CRITICAL   = ["bankruptcy", "insider trading", "sec investigation", "fda approval", "data breach", "class action", "restructuring", "failure to file"]
BASE_STRONG     = ["upgrade", "beat estimates", "record", "surge", "outperform", "raise", "warning", "cut"]
BASE_MODERATE   = ["buy", "positive", "growth", "profit", "decline", "drop", "loss"]
BASE_PRIORITY   = ["earnings", "downgrade", "price target", "miss estimates", "guidance", "dividend", "buyback", "merger", "acquisition", "ipo", "layoff", "revenue"]

def expand_synonyms(words):
    syns = set(words)
    for w in words:
        for synset in wordnet.synsets(w):
            for lemma in synset.lemmas():
                syns.add(lemma.name().lower().replace('_', ' '))
    return list(syns)

CRITICAL_KEYWORDS = expand_synonyms(BASE_CRITICAL)
STRONG_KEYWORDS   = expand_synonyms(BASE_STRONG)
MODERATE_KEYWORDS = expand_synonyms(BASE_MODERATE)
PRIORITY_KEYWORDS = expand_synonyms(BASE_PRIORITY)

# === ML MODEL SETUP ===
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
model      = LogisticRegression()
model_lock = threading.Lock()

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

# === FINBERT SETUP ===
finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model     = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def analyze_sentiment(text: str) -> float:
    inputs  = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = finbert_model(**inputs)
    probs   = F.softmax(outputs.logits, dim=1).detach().numpy()[0]
    return probs[2] - probs[0]

def get_sentiment_label(score: float) -> str:
    if score > 0.05:
        return "Bullish"
    elif score < -0.05:
        return "Bearish"
    else:
        return "Neutral"

def get_vix_level():
    try:
        hist   = yf.Ticker("^VIX").history(period="1d", interval="1m")
        latest = hist["Close"].iloc[-1]
        if latest < 14:     lbl = "🟢 Low Fear"
        elif latest < 20:   lbl = "🟡 Normal"
        elif latest < 25:   lbl = "🟠 Caution"
        elif latest < 30:   lbl = "🔴 High Fear"
        else:               lbl = "🚨 Panic"
        return round(latest,2), lbl
    except Exception as e:
        return None, f"❌ VIX error: {e}"

def send_to_telegram(message: str):
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=5)
        resp.raise_for_status()
        print(f"✅ Telegram: {resp.status_code}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def is_rate_limited(ticker: str) -> bool:
    last = rate_limit_data.get(ticker, 0)
    return (time.time() - last) < RATE_LIMIT_SECONDS

def update_rate_limit(ticker: str):
    rate_limit_data[ticker] = time.time()
    RATE_LIMIT_LOG_PATH.write_text(json.dumps(rate_limit_data), encoding="utf-8")

def match_watchlist(text: str) -> str:
    tl = text.lower()
    for ticker, kws in WATCHLIST.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw.lower())}\b", tl):
                return ticker
    return "GENERAL"

def calculate_confidence(headline: str) -> (int, str):
    hl = headline.lower()
    hi = sum(w in hl for w in CRITICAL_KEYWORDS)
    st = sum(w in hl for w in STRONG_KEYWORDS)
    md = sum(w in hl for w in MODERATE_KEYWORDS)
    pr = sum(w in hl for w in PRIORITY_KEYWORDS)
    score = min(100, hi*30 + st*20 + md*10 + pr*5)
    label = "High" if score >= 80 else "Medium" if score >= CONFIDENCE_THRESHOLD else "Low"
    return score, label

def should_send_alert(title: str, ticker: str, conf_score: int) -> bool:
    if title in sent_news or is_rate_limited(ticker):
        return False
    if ticker == "GENERAL" and conf_score < CONFIDENCE_THRESHOLD:
        return False
    return True

def send_alert(title: str, ticker: str, sentiment: float, conf_score: int, conf_label: str, source: str):
    sentiment_label = get_sentiment_label(sentiment)
    vix_val, vix_lbl = get_vix_level()
    ml_pred, ml_conf = classify_text(title)
    timestamp        = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = (
        f"🗞 *{source} Alert*\n"
        f"*{ticker}* — {title}\n"
        f"📈 Sentiment: *{sentiment_label}* (`{sentiment:.2f}`)\n"
        f"🎯 Confidence: *{conf_score}%* ({conf_label})"
    )
    if ml_pred:
        msg += f"\n🤖 ML: *{ml_pred}* ({ml_conf}%)"
    msg += f"\n🌪 VIX: *{vix_val}* — {vix_lbl}  🕒 {timestamp}"
    send_to_telegram(msg)
    sent_news.add(title)
    SENT_LOG_PATH.write_text("\n".join(sent_news), encoding="utf-8")
    update_rate_limit(ticker)

def process_yahoo_entry(entry):
    title = entry.get("title", "").strip()
    print("▶️ Yahoo headline:", title)
    ticker = match_watchlist(title)
    conf_score, conf_label = calculate_confidence(title)
    print(f"   → ticker: {ticker}  │ conf_score: {conf_score}% ({conf_label})")
    if should_send_alert(title, ticker, conf_score):
        print("   → passing filters, sending alert")
        sentiment = analyze_sentiment(title)
        send_alert(title, ticker, sentiment, conf_score, conf_label, "Yahoo")
    else:
        print("   → filtered out, not sending")

def analyze_yahoo():
    print("📡 Scanning Yahoo RSS...")
    feed = feedparser.parse(RSS_URL)
    for entry in feed.entries:
        process_yahoo_entry(entry)
    print("✅ Yahoo done.")

def fetch_benzinga(chunk):
    try:
        resp = requests.get(
            BENZINGA_URL,
            params={"tickers": ",".join(chunk), "items": 50, "token": BENZINGA_API_KEY},
            timeout=10
        )
        print("🔍 Benzinga HTTP:", resp.status_code, resp.text[:100])
        resp.raise_for_status()
        return resp.json().get("news", [])
    except ValueError as e:
        print("❌ Benzinga JSON error:", e)
        return []
    except Exception as e:
        print(f"❌ Benzinga error: {e}")
        return []

def process_benzinga_article(article):
    title = article.get("title", "").strip()
    print("▶️ Benzinga headline:", title)
    ticker = match_watchlist(title)
    conf_score, conf_label = calculate_confidence(title)
    print(f"   → ticker: {ticker}  │ conf_score: {conf_score}% ({conf_label})")
    if should_send_alert(title, ticker, conf_score):
        print("   → passing filters, sending alert")
        sentiment = analyze_sentiment(title)
        send_alert(title, ticker, sentiment, conf_score, conf_label, "Benzinga")
    else:
        print("   → filtered out, not sending")

def analyze_benzinga():
    print("📡 Scanning Benzinga...")
    symbols = list(WATCHLIST.keys())
    for i in range(0, len(symbols), 20):
        chunk = symbols[i:i+20]
        for article in fetch_benzinga(chunk):
            process_benzinga_article(article)
    print("✅ Benzinga done.")

if __name__ == "__main__":
    print("🚀 Starting market bot...")
    # Load persistent data
    sent_news       = set(SENT_LOG_PATH.read_text(encoding="utf-8").splitlines()) if SENT_LOG_PATH.exists() else set()
    rate_limit_data = json.loads(RATE_LIMIT_LOG_PATH.read_text(encoding="utf-8")) if RATE_LIMIT_LOG_PATH.exists() else {}
    training_data   = json.loads(TRAINING_DATA_PATH.read_text(encoding="utf-8")) if TRAINING_DATA_PATH.exists() else {"texts": [], "labels": []}

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
