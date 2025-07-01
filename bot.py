import requests
import feedparser
from pathlib import Path
import time
import re
import json
import threading
from itertools import islice

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# FinBERT imports
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M"
TELEGRAM_CHAT_ID = "5528794335"

# === RSS & BENZINGA CONFIG ===
RSS_URL = "https://finance.yahoo.com/rss/topstories"
BENZINGA_API_KEY = "bz.XAO6BCTUMYPFGHXXL7SJ3ZU4IRRTFRE7"
BENZINGA_URL = "https://api.benzinga.com/api/v2/news"

# === THRESHOLDS & LIMITS ===
SENTIMENT_THRESHOLD = 0.2    # FinBERT threshold
CONFIDENCE_THRESHOLD = 30    # Minimum confidence % to alert
RATE_LIMIT_SECONDS = 1800    # 30-minute cooldown per ticker

# === LOG FILE PATHS ===
SENT_LOG_PATH = Path("sent_titles.txt")
FEEDBACK_LOG_PATH = Path("feedback.json")
RATE_LIMIT_LOG_PATH = Path("rate_limit.json")
TRAINING_DATA_PATH = Path("training_data.json")

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
    "QCOM": ["QCOM", "Qualcomm"],
}

# === Initialize persistent storage ===
sent_news = set(SENT_LOG_PATH.read_text(encoding="utf-8").splitlines()) if SENT_LOG_PATH.exists() else set()
rate_limit_data = json.loads(RATE_LIMIT_LOG_PATH.read_text(encoding="utf-8")) if RATE_LIMIT_LOG_PATH.exists() else {}
feedback_data = json.loads(FEEDBACK_LOG_PATH.read_text(encoding="utf-8")) if FEEDBACK_LOG_PATH.exists() else {}
training_data = json.loads(TRAINING_DATA_PATH.read_text(encoding="utf-8")) if TRAINING_DATA_PATH.exists() else {"texts": [], "labels": []}

# === ML Model Setup ===
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
model = LogisticRegression()
model_lock = threading.Lock()

# === FinBERT Setup ===
finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# === Confidence Keywords ===
PRIORITY_KEYWORDS = [
    "earnings", "upgrade", "downgrade", "price target", "beat estimates",
    "miss estimates", "warning", "lawsuit", "guidance", "dividend",
    "buyback", "merger", "acquisition", "ipo", "layoff",
    "revenue", "profit"
]

# === Utility Functions ===
def calculate_confidence(headline: str) -> (int, str):
    hl = headline.lower()
    strong = ["upgrade", "beat estimates", "record", "surge", "outperform", "raise", "warning", "cut"]
    moderate = ["buy", "positive", "growth", "profit", "decline", "drop", "loss"]
    priority = PRIORITY_KEYWORDS

    s_hits = sum(w in hl for w in strong)
    m_hits = sum(w in hl for w in moderate)
    p_hits = sum(w in hl for w in priority)
    score = min(100, s_hits*25 + m_hits*10 + p_hits*15)
    label = "High" if score >= 70 else "Medium" if score >= CONFIDENCE_THRESHOLD else "Low"
    return score, label


def send_to_telegram(message: str):
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=5)
        print("✅ Telegram alert sent.")
    except Exception as e:
        print(f"❌ Telegram error: {e}")


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


def is_rate_limited(ticker: str) -> bool:
    last = rate_limit_data.get(ticker, 0)
    if time.time() - last < RATE_LIMIT_SECONDS:
        print(f"⏳ Rate limited: {ticker}")
        return True
    return False


def update_rate_limit(ticker: str):
    rate_limit_data[ticker] = time.time()
    RATE_LIMIT_LOG_PATH.write_text(json.dumps(rate_limit_data), encoding="utf-8")

# === Yahoo Helpers ===
def process_yahoo_entry(entry):
    title = entry.get("title", "").strip()
    if not title:
        return
    ticker = match_watchlist(title)
    if not ticker:
        return
    sentiment = analyze_sentiment(title)
    conf_score, conf_label = calculate_confidence(title)
    if should_send_alert(title, ticker, sentiment, conf_score):
        send_alert(title, ticker, sentiment, conf_score, conf_label, source="Yahoo")


def analyze_yahoo():
    print("📡 Scanning Yahoo RSS...")
    feed = feedparser.parse(RSS_URL)
    for entry in feed.entries:
        process_yahoo_entry(entry)
    print("✅ Yahoo done.")

# === Benzinga Helpers ===
def fetch_benzinga(chunk):
    try:
        resp = requests.get(BENZINGA_URL, params={"tickers": ",".join(chunk), "items":50, "token":BENZINGA_API_KEY}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("news", [])
    except Exception as e:
        print(f"❌ Benzinga error: {e}")
        return []


def process_benzinga_article(article):
    title = article.get("title", "").strip()
    if not title:
        return
    ticker = match_watchlist(title)
    if not ticker:
        return
    sentiment = analyze_sentiment(title)
    conf_score, conf_label = calculate_confidence(title)
    if should_send_alert(title, ticker, sentiment, conf_score):
        send_alert(title, ticker, sentiment, conf_score, conf_label, source="Benzinga")


def analyze_benzinga():
    print("📡 Scanning Benzinga...")

