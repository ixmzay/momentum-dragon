import os
import time
import re
import json
import threading
from pathlib import Path
from datetime import datetime

import requests
import feedparser
import yfinance as yf
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

import nltk
from nltk.corpus import wordnet

# Ensure WordNet is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# === TELEGRAM & BENZINGA KEYS ===
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5528794335")
BENZINGA_API_KEY = os.getenv("BENZINGA_API_KEY", "bz.XAO6BCTUMYPFGHXXL7SJ3ZU4IRRTFRE7")

# === URLS & THRESHOLDS ===
RSS_URL              = "https://finance.yahoo.com/rss/topstories"
BENZINGA_URL         = "https://api.benzinga.com/api/v2/news"
CONFIDENCE_THRESHOLD = 60    # Keyword-based medium cutoff
NET_THRESHOLD        = 0.2   # FinBERT net-sentiment threshold for GENERAL
RATE_LIMIT_SECONDS   = 1800  # 30-minute cooldown per ticker

# === LOG / DATA PATHS ===
SENT_LOG_PATH       = Path("sent_titles.txt")
RATE_LIMIT_LOG_PATH = Path("rate_limit.json")
TRAINING_DATA_PATH  = Path("training_data.json")
LAST_RUN_PATH       = Path("last_run.json")

# === GLOBAL STATE ===
sent_news       = set()
rate_limit_data = {}
training_data   = {"texts": [], "labels": []}
last_run_ts     = 0.0

# === FULL WATCHLIST ===
WATCHLIST = {
    "AAPL": ["AAPL","Apple"],
    "MSFT": ["MSFT","Microsoft"],
    "GOOGL": ["GOOGL","Google","Alphabet"],
    "AMZN": ["AMZN","Amazon"],
    "META": ["META","Facebook","Meta"],
    "TSLA": ["TSLA","Tesla"],
    "NVDA": ["NVDA","NVIDIA"],
    "AMD": ["AMD","Advanced Micro Devices"],
    "INTC": ["INTC","Intel"],
    "NFLX": ["NFLX","Netflix"],
    "SPY": ["SPY","S&P 500"],
    "QQQ": ["QQQ","Nasdaq"],
    "IWM": ["IWM","Russell 2000"],
    "XOM": ["XOM","Exxon","ExxonMobil"],
    "CVX": ["CVX","Chevron"],
    "OXY": ["OXY","Occidental"],
    "WMT": ["WMT","Walmart"],
    "COST": ["COST","Costco"],
    "TGT": ["TGT","Target"],
    "HD": ["HD","Home Depot"],
    "LOW": ["LOW","Lowe's"],
    "JPM": ["JPM","JPMorgan"],
    "BAC": ["BAC","Bank of America"],
    "GS": ["GS","Goldman Sachs"],
    "MS": ["MS","Morgan Stanley"],
    "WFC": ["WFC","Wells Fargo"],
    "BX": ["BX","Blackstone"],
    "UBER": ["UBER","Uber"],
    "LYFT": ["LYFT","Lyft"],
    "SNOW": ["SNOW","Snowflake"],
    "PLTR": ["PLTR","Palantir"],
    "CRM": ["CRM","Salesforce"],
    "ADBE": ["ADBE","Adobe"],
    "SHOP": ["SHOP","Shopify"],
    "PYPL": ["PYPL","PayPal"],
    "SQ": ["SQ","Block"],
    "COIN": ["COIN","Coinbase"],
    "ROKU": ["ROKU","Roku"],
    "BABA": ["BABA","Alibaba"],
    "JD": ["JD","JD.com"],
    "NIO": ["NIO"],
    "LI": ["LI","Li Auto"],
    "XPEV": ["XPEV","XPeng"],
    "LMT": ["LMT","Lockheed Martin"],
    "NOC": ["NOC","Northrop Grumman"],
    "RTX": ["RTX","Raytheon"],
    "BA": ["BA","Boeing"],
    "GE": ["GE","General Electric"],
    "CAT": ["CAT","Caterpillar"],
    "DE": ["DE","John Deere"],
    "F": ["F","Ford"],
    "GM": ["GM","General Motors"],
    "RIVN": ["RIVN","Rivian"],
    "LCID": ["LCID","Lucid"],
    "PFE": ["PFE","Pfizer"],
    "MRNA": ["MRNA","Moderna"],
    "JNJ": ["JNJ","Johnson & Johnson"],
    "BMY": ["BMY","Bristol Myers"],
    "UNH": ["UNH","UnitedHealth"],
    "MDT": ["MDT","Medtronic"],
    "ABBV": ["ABBV","AbbVie"],
    "TMO": ["TMO","Thermo Fisher"],
    "SHEL": ["SHEL","Shell"],
    "BP": ["BP","British Petroleum"],
    "UL": ["UL","Unilever"],
    "BTI": ["BTI","British American Tobacco"],
    "SAN": ["SAN","Santander"],
    "DB": ["DB","Deutsche Bank"],
    "VTOL": ["VTOL","Bristow Group"],
    "EVTL": ["EVTL","Vertical Aerospace"],
    "EH": ["EH","EHang"],
    "PL": ["PL","Planet Labs"],
    "TT": ["TT","Trane"],
    "JCI": ["JCI","Johnson Controls"],
    "RDW": ["RDW","Redwire"],
    "LOAR": ["LOAR","Loar Holdings"],
    "PANW": ["PANW","Palo Alto Networks"],
    "CRWD": ["CRWD","CrowdStrike"],
    "NET": ["NET","Cloudflare"],
    "ZS": ["ZS","Zscaler"],
    "TSM": ["TSM","Taiwan Semiconductor"],
    "AVGO": ["AVGO","Broadcom"],
    "MU": ["MU","Micron"],
    "TXN": ["TXN","Texas Instruments"],
    "QCOM": ["QCOM","Qualcomm"],
    "NKE": ["NKE","Nike"],
}

# === OVERRIDES & KEYWORDS ===
BULLISH_OVERRIDES = [
    "dividend","buyback","upgrade","beat estimates","raise","surge",
    "outperform","jump","gain","rise","rally","merge","acquisition","headed"
]
BEARISH_OVERRIDES = [
    "downgrade","miss estimates","warning","cut","plunge","crash",
    "selloff","fall","decline","drop","slump","bankruptcy"
]

BASE_CRITICAL   = [
    "bankruptcy","insider trading","sec investigation","fda approval",
    "data breach","class action","restructuring","failure to file"
]
BASE_STRONG     = [
    "upgrade","beat estimates","record","surge","outperform","raise","warning","cut"
]
BASE_MODERATE   = ["buy","positive","growth","profit","decline","drop","loss"]
BASE_PRIORITY   = [
    "earnings","downgrade","price target","miss estimates","guidance",
    "dividend","buyback","merger","acquisition","ipo","layoff","revenue"
]

def expand_synonyms(words):
    syns = set(words)
    for w in words:
        for synset in wordnet.synsets(w):
            for lemma in synset.lemmas():
                syns.add(lemma.name().lower().replace('_',' '))
    return list(syns)

CRITICAL_KEYWORDS = expand_synonyms(BASE_CRITICAL)
STRONG_KEYWORDS   = expand_synonyms(BASE_STRONG)
MODERATE_KEYWORDS = expand_synonyms(BASE_MODERATE)
PRIORITY_KEYWORDS = expand_synonyms(BASE_PRIORITY)

# === ML SETUP ===
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
model      = LogisticRegression()
model_lock = threading.Lock()

def suggest_confidence_threshold(pct=0.75):
    scores = [calculate_confidence(t)[0] for t in training_data.get("texts", [])]
    if not scores:
        print("⚠️ No data to suggest threshold.")
        return
    scores.sort()
    idx = int(len(scores) * pct)
    print(f"💡 {int(pct*100)}th percentile keyword score: {scores[min(idx,len(scores)-1)]}")

def train_model():
    texts  = training_data.get("texts", [])
    labels = training_data.get("labels", [])
    if len(texts) < 10:
        print("⚠️ Not enough training data.")
        return
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

def classify_text(text: str):
    if not training_data.get("texts"):
        return None, None
    X    = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max() * 100
    return pred, round(prob,2)

# === FINBERT SETUP ===
finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model     = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def analyze_sentiment_probs(text: str):
    inp = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    out = finbert_model(**inp)
    return F.softmax(out.logits, dim=1).detach().numpy()[0]

def analyze_sentiment_net(text: str) -> float:
    p = analyze_sentiment_probs(text)
    return float(p[2] - p[0])

def analyze_sentiment_argmax(text: str) -> str:
    p = analyze_sentiment_probs(text)
    return ["Bearish","Neutral","Bullish"][int(p.argmax())]

def get_sentiment_label(text: str) -> str:
    t = text.strip().lower()
    if t.endswith("?"):
        return "Neutral"
    if any(kw in t for kw in BULLISH_OVERRIDES):
        return "Bullish"
    if any(kw in t for kw in BEARISH_OVERRIDES):
        return "Bearish"
    return analyze_sentiment_argmax(text)

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

def send_to_telegram(msg: str):
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        print(f"✅ Telegram: {r.status_code}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def is_rate_limited(ticker: str) -> bool:
    last = rate_limit_data.get(ticker,0)
    return (time.time() - last) < RATE_LIMIT_SECONDS

def update_rate_limit(ticker: str):
    rate_limit_data[ticker] = time.time()
    RATE_LIMIT_LOG_PATH.write_text(json.dumps(rate_limit_data), encoding="utf-8")

# === TICKER VALIDATION & FINDER ===
_ticker_cache: dict[str,bool] = {}

def is_valid_ticker(sym: str) -> bool:
    sym = sym.upper()
    if sym in _ticker_cache:
        return _ticker_cache[sym]
    try:
        info = yf.Ticker(sym).info
        valid = info.get("regularMarketPrice") is not None
    except Exception:
        valid = False
    _ticker_cache[sym] = valid
    return valid

def match_watchlist_alias(text: str) -> str | None:
    lo = text.lower()
    for sym, aliases in WATCHLIST.items():
        for kw in aliases:
            if kw.lower() in lo:
                return sym
    return None

def find_ticker_in_text(text: str) -> str | None:
    up = text.upper()
    candidates = set()
    candidates.update(re.findall(r'\$([A-Z]{1,5})\b', up))
    candidates.update(re.findall(r'\(([A-Z]{1,5})\)', up))
    candidates.update(re.findall(r'\b([A-Z]{1,5})\b', up))
    for sym in candidates:
        if is_valid_ticker(sym):
            return sym
    return None

def calculate_confidence(headline: str) -> (int,str):
    hl = headline.lower()
    hi = sum(w in hl for w in CRITICAL_KEYWORDS)
    st = sum(w in hl for w in STRONG_KEYWORDS)
    md = sum(w in hl for w in MODERATE_KEYWORDS)
    pr = sum(w in hl for w in PRIORITY_KEYWORDS)
    score = min(100, hi*30 + st*20 + md*10 + pr*5)
    if score >= 80:
        lbl = "High"
    elif score >= CONFIDENCE_THRESHOLD:
        lbl = "Medium"
    else:
        lbl = "Low"
    return score, lbl

def load_last_run() -> float:
    return float(LAST_RUN_PATH.read_text()) if LAST_RUN_PATH.exists() else 0.0

def save_last_run(ts: float):
    LAST_RUN_PATH.write_text(str(ts), encoding="utf-8")

def record_feedback(title: str, label: str):
    training_data.setdefault("texts", []).append(title)
    training_data.setdefault("labels", []).append(label)
    TRAINING_DATA_PATH.write_text(json.dumps(training_data), encoding="utf-8")

def should_send_alert(title: str, ticker: str, conf: int, summary: str) -> bool:
    if title in sent_news or is_rate_limited(ticker):
        return False
    if ticker != "GENERAL":
        return True
    net = analyze_sentiment_net(summary)
    return conf >= CONFIDENCE_THRESHOLD or abs(net) >= NET_THRESHOLD

def send_alert(title: str, ticker: str, summary: str,
               conf: int, conf_lbl: str, source: str):
    net      = analyze_sentiment_net(summary)
    sent_lbl = get_sentiment_label(summary)
    vix_val, vix_lbl = get_vix_level()
    ml_pred, ml_conf = classify_text(title)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        f"🗞 *{source} Alert*",
        f"*{ticker}* — {title}",
        f"📈 Sentiment: *{sent_lbl}* (`{net:.2f}`)",
        f"🎯 Confidence: *{conf}%* ({conf_lbl})"
    ]
    if ml_pred:
        lines.append(f"🤖 ML: *{ml_pred}* ({ml_conf}%)")
    lines.append(f"🌪 VIX: *{vix_val}* — {vix_lbl}  🕒 {ts}")

    send_to_telegram("\n".join(lines))
    sent_news.add(title)
    SENT_LOG_PATH.write_text("\n".join(sent_news), encoding="utf-8")
    update_rate_limit(ticker)

def fetch_article_content(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return " ".join(p.get_text() for p in soup.find_all("p"))
    except Exception as e:
        print(f"❌ Article fetch error: {e}")
        return ""

def process_yahoo_entry(entry):
    global last_run_ts
    if hasattr(entry, "published_parsed"):
        ts = time.mktime(entry.published_parsed)
        if ts <= last_run_ts:
            return
        last_run_ts = max(last_run_ts, ts)

    title   = entry.get("title","").strip()
    summary = getattr(entry,"summary_detail",{}).get("value","") or entry.get("summary","")
    text    = BeautifulSoup(summary or title, "html.parser").get_text().strip()

    print("▶️ Yahoo headline:", title)
    ticker = match_watchlist_alias(title) or find_ticker_in_text(title) or "GENERAL"
    print(f"   → ticker: {ticker}")

    conf, conf_lbl = calculate_confidence(title)
    if should_send_alert(title, ticker, conf, text):
        print("   → sending alert")
        send_alert(title, ticker, text, conf, conf_lbl, "Yahoo")
    else:
        print("   → filtered out")

def analyze_yahoo():
    global last_run_ts
    last_run_ts = load_last_run()
    print("📡 Scanning Yahoo RSS...")
    feed = feedparser.parse(RSS_URL)
    for e in feed.entries:
        process_yahoo_entry(e)
    save_last_run(last_run_ts)
    print("✅ Yahoo done.")

def process_benzinga_article(a):
    title = a.get("title","").strip()
    url   = a.get("url") or a.get("sourceUrl","")
    text  = fetch_article_content(url) if url else title

    print("▶️ Benzinga headline:", title)
    ticker = match_watchlist_alias(title) or find_ticker_in_text(title) or "GENERAL"
    print(f"   → ticker: {ticker}")

    conf, conf_lbl = calculate_confidence(title)
    if should_send_alert(title, ticker, conf, text):
        print("   → sending alert")
        send_alert(title, ticker, text, conf, conf_lbl, "Benzinga")
    else:
        print("   → filtered out")

def analyze_benzinga():
    print("📡 Scanning Benzinga...")
    symbols = list(WATCHLIST.keys())
    for i in range(0, len(symbols), 20):
        for art in fetch_benzinga(symbols[i:i+20]):
            process_benzinga_article(art)
    print("✅ Benzinga done.")

def fetch_benzinga(chunk):
    try:
        r = requests.get(BENZINGA_URL, params={
            "tickers": ",".join(chunk),
            "items": 50,
            "token": BENZINGA_API_KEY
        }, timeout=10)
        r.raise_for_status()
        return r.json().get("news", [])
    except Exception as e:
        print(f"❌ Benzinga error: {e}")
        return []

if __name__ == "__main__":
    sent_news       = set(SENT_LOG_PATH.read_text().splitlines()) if SENT_LOG_PATH.exists() else set()
    rate_limit_data = json.loads(RATE_LIMIT_LOG_PATH.read_text()) if RATE_LIMIT_LOG_PATH.exists() else {}
    training_data   = json.loads(TRAINING_DATA_PATH.read_text()) if TRAINING_DATA_PATH.exists() else {"texts":[],"labels":[]}
    last_run_ts     = load_last_run()

    print("🚀 Starting market bot…")
    suggest_confidence_threshold(0.75)
    train_model()

    while True:
        try:
            analyze_yahoo()
            analyze_benzinga()
            print("⏲ Sleeping 60s…\n")
            time.sleep(60)
        except Exception as e:
            print(f"💥 Main loop error: {e}")
            time.sleep(10)
