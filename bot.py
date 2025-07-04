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

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN   = "YOUR_TELEGRAM_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# === RSS & BENZINGA CONFIG ===
RSS_URL          = "https://finance.yahoo.com/rss/topstories"
BENZINGA_API_KEY = "YOUR_BENZINGA_KEY"
BENZINGA_URL     = "https://api.benzinga.com/api/v2/news"

# === THRESHOLDS & LIMITS ===
CONFIDENCE_THRESHOLD = 60    # keyword-based medium cutoff
NET_THRESHOLD        = 0.2   # FinBERT net-sentiment threshold for GENERAL
RATE_LIMIT_SECONDS   = 1800  # 30-minute cooldown per ticker

# === LOG FILE PATHS ===
SENT_LOG_PATH       = Path("sent_titles.txt")
RATE_LIMIT_LOG_PATH = Path("rate_limit.json")
TRAINING_DATA_PATH  = Path("training_data.json")
LAST_RUN_PATH       = Path("last_run.json")

# === GLOBAL STORAGE ===
sent_news       = set()
rate_limit_data = {}
training_data   = {"texts": [], "labels": []}
last_run_ts     = 0.0

# === WATCHLIST (can be extended at will) ===
WATCHLIST = {
    "AAPL": ["Apple"],
    "MSFT": ["Microsoft"],
    "GOOGL": ["Google", "Alphabet"],
    "AMZN": ["Amazon"],
    "META": ["Facebook", "Meta"],
    "TSLA": ["Tesla"],
    "NVDA": ["NVIDIA"],
    "AMD": ["Advanced Micro Devices"],
    "INTC": ["Intel"],
    "NFLX": ["Netflix"],
    "SPY": ["S&P 500"],
    "QQQ": ["Nasdaq"],
    "IWM": ["Russell 2000"],
    "XOM": ["Exxon", "ExxonMobil"],
    "CVX": ["Chevron"],
    "OXY": ["Occidental"],
    "WMT": ["Walmart"],
    "COST": ["Costco"],
    "TGT": ["Target"],
    "HD": ["Home Depot"],
    "LOW": ["Lowe's"],
    "JPM": ["JPMorgan"],
    "BAC": ["Bank of America"],
    "GS": ["Goldman Sachs"],
    "MS": ["Morgan Stanley"],
    "WFC": ["Wells Fargo"],
    "BX": ["Blackstone"],
    "UBER": ["Uber"],
    "LYFT": ["Lyft"],
    "SNOW": ["Snowflake"],
    "PLTR": ["Palantir"],
    "CRM": ["Salesforce"],
    "ADBE": ["Adobe"],
    "SHOP": ["Shopify"],
    "PYPL": ["PayPal"],
    "SQ": ["Block"],
    "COIN": ["Coinbase"],
    "ROKU": ["Roku"],
    "BABA": ["Alibaba"],
    "JD": ["JD.com"],
    "NIO": ["NIO"],
    "LI": ["Li Auto"],
    "XPEV": ["XPeng"],
    "LMT": ["Lockheed Martin"],
    "NOC": ["Northrop Grumman"],
    "RTX": ["Raytheon"],
    "BA": ["Boeing"],
    "GE": ["General Electric"],
    "CAT": ["Caterpillar"],
    "DE": ["John Deere"],
    "F": ["Ford"],
    "GM": ["General Motors"],
    "RIVN": ["Rivian"],
    "LCID": ["Lucid"],
    "PFE": ["Pfizer"],
    "MRNA": ["Moderna"],
    "JNJ": ["Johnson & Johnson"],
    "BMY": ["Bristol Myers"],
    "UNH": ["UnitedHealth"],
    "MDT": ["Medtronic"],
    "ABBV": ["AbbVie"],
    "TMO": ["Thermo Fisher"],
    "SHEL": ["Shell"],
    "BP": ["British Petroleum"],
    "UL": ["Unilever"],
    "BTI": ["British American Tobacco"],
    "SAN": ["Santander"],
    "DB": ["Deutsche Bank"],
    "VTOL": ["Bristow Group"],
    "EVTL": ["Vertical Aerospace"],
    "EH": ["EHang"],
    "PL": ["Planet Labs"],
    "TT": ["Trane"],
    "JCI": ["Johnson Controls"],
    "RDW": ["Redwire"],
    "LOAR": ["Loar Holdings"],
    "PANW": ["Palo Alto Networks"],
    "CRWD": ["CrowdStrike"],
    "NET": ["Cloudflare"],
    "ZS": ["Zscaler"],
    "TSM": ["Taiwan Semiconductor"],
    "AVGO": ["Broadcom"],
    "MU": ["Micron"],
    "TXN": ["Texas Instruments"],
    "QCOM": ["Qualcomm"],
    "NKE": ["Nike"]
}

# === OVERRIDE LISTS ===
BULLISH_OVERRIDES = [
    "dividend","buyback","upgrade","beat estimates","raise","surge",
    "outperform","jump","jumps","gain","gains","rise","rises","soar",
    "soars","rally","rallies","higher","merge","merges","merger",
    "acquisition","rated buy","headed"
]
BEARISH_OVERRIDES = [
    "downgrade","miss estimates","warning","cut","plunge","plunges",
    "crash","crashes","selloff","sell-off","fall","falls","decline",
    "declines","drop","drops","slump","slumps","bankruptcy"
]

# === BASE KEYWORDS & SYNONYMS ===
BASE_CRITICAL   = ["bankruptcy","insider trading","sec investigation",
                   "fda approval","data breach","class action",
                   "restructuring","failure to file"]
BASE_STRONG     = ["upgrade","beat estimates","record","surge",
                   "outperform","raise","warning","cut"]
BASE_MODERATE   = ["buy","positive","growth","profit","decline",
                   "drop","loss"]
BASE_PRIORITY   = ["earnings","downgrade","price target","miss estimates",
                   "guidance","dividend","buyback","merger","acquisition",
                   "ipo","layoff","revenue"]

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
    return pred, round(prob, 2)

def suggest_confidence_threshold(pct=0.75):
    scores = []
    for t in training_data.get("texts", []):
        s,_ = calculate_confidence(t)
        scores.append(s)
    if not scores:
        print("⚠️ No training data to suggest threshold.")
        return
    scores.sort()
    idx = int(len(scores) * pct)
    sug = scores[min(idx, len(scores)-1)]
    print(f"💡 {int(pct*100)}th percentile keyword score is {sug}. Consider CONFIDENCE_THRESHOLD={sug}")

# === FINBERT SETUP ===
finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model     = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def analyze_sentiment_probs(text: str):
    inputs  = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = finbert_model(**inputs)
    return F.softmax(outputs.logits, dim=1).detach().numpy()[0]

def analyze_sentiment_net(text: str) -> float:
    probs = analyze_sentiment_probs(text)
    return probs[2] - probs[0]

def analyze_sentiment_argmax(text: str) -> str:
    probs = analyze_sentiment_probs(text)
    return ["Bearish","Neutral","Bullish"][int(probs.argmax())]

def get_sentiment_label(text: str) -> str:
    txt = text.strip().lower()
    if txt.endswith("?"):
        return "Neutral"
    if any(kw in txt for kw in BULLISH_OVERRIDES):
        return "Bullish"
    if any(kw in txt for kw in BEARISH_OVERRIDES):
        return "Bearish"
    return analyze_sentiment_argmax(text)

def get_vix_level():
    try:
        hist   = yf.Ticker("^VIX").history(period="1d", interval="1m")
        latest = hist["Close"].iloc[-1]
        if latest < 14:
            lbl = "🟢 Low Fear"
        elif latest < 20:
            lbl = "🟡 Normal"
        elif latest < 25:
            lbl = "🟠 Caution"
        elif latest < 30:
            lbl = "🔴 High Fear"
        else:
            lbl = "🚨 Panic"
        return round(latest, 2), lbl
    except Exception as e:
        return None, f"❌ VIX error: {e}"

def send_to_telegram(message: str):
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json=payload,
            timeout=10
        )
        r.raise_for_status()
        print(f"✅ Telegram: {r.status_code}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def is_rate_limited(ticker: str) -> bool:
    last = rate_limit_data.get(ticker, 0)
    return (time.time() - last) < RATE_LIMIT_SECONDS

def update_rate_limit(ticker: str):
    rate_limit_data[ticker] = time.time()
    RATE_LIMIT_LOG_PATH.write_text(json.dumps(rate_limit_data), encoding="utf-8")

TICKER_REGEX = re.compile(r'\$([A-Z]{1,5})\b|\(([A-Z]{1,5})\)')

def match_watchlist_alias(text: str) -> str | None:
    lo = text.lower()
    for t, kws in WATCHLIST.items():
        for kw in kws:
            if kw.lower() in lo:
                return t
    return None

def find_ticker_in_text(text: str) -> str | None:
    up = text.upper()
    # $SYM
    m = re.search(r'\$([A-Z]{1,5})\b', up)
    if m:
        return m.group(1)
    # (SYM)
    m = re.search(r'\(([A-Z]{1,5})\)', up)
    if m:
        return m.group(1)
    # bare ALL-CAPS words
    for w in up.split():
        if re.fullmatch(r'[A-Z]{1,5}', w):
            return w
    return None

def calculate_confidence(headline: str) -> (int, str):
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
    if LAST_RUN_PATH.exists():
        return float(LAST_RUN_PATH.read_text())
    return 0.0

def save_last_run(ts: float):
    LAST_RUN_PATH.write_text(str(ts))#endregion

def record_feedback(title: str, label: str):
    training_data.setdefault("texts", []).append(title)
    training_data.setdefault("labels", []).append(label)
    TRAINING_DATA_PATH.write_text(json.dumps(training_data), encoding="utf-8")

def should_send_alert(title: str, ticker: str, conf: int, summary: str) -> bool:
    if title in sent_news or is_rate_limited(ticker):
        return False
    # always send if ticker identified
    if ticker.upper() != "GENERAL":
        return True
    net = analyze_sentiment_net(summary)
    return conf >= CONFIDENCE_THRESHOLD or abs(net) >= NET_THRESHOLD

def send_alert(title: str, ticker: str, summary: str,
               conf: int, conf_lbl: str, source: str):
    net     = analyze_sentiment_net(summary)
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
    # skip old items
    if hasattr(entry, "published_parsed"):
        ts = time.mktime(entry.published_parsed)
        if ts <= last_run_ts:
            return
        last_run_ts = max(last_run_ts, ts)

    title   = entry.get("title", "").strip()
    summary = getattr(entry, "summary_detail", {}).get("value", "") or entry.get("summary", "")
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
    title = a.get("title", "").strip()
    url   = a.get("url") or a.get("sourceUrl", "")
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
        r = requests.get(
            BENZINGA_URL,
            params={"tickers": ",".join(chunk), "items": 50, "token": BENZINGA_API_KEY},
            timeout=10
        )
        r.raise_for_status()
        return r.json().get("news", [])
    except Exception as e:
        print(f"❌ Benzinga error: {e}")
        return []

if __name__ == "__main__":
    sent_news       = set(SENT_LOG_PATH.read_text(encoding="utf-8").splitlines()) \
                        if SENT_LOG_PATH.exists() else set()
    rate_limit_data = json.loads(RATE_LIMIT_LOG_PATH.read_text(encoding="utf-8")) \
                        if RATE_LIMIT_LOG_PATH.exists() else {}
    training_data   = json.loads(TRAINING_DATA_PATH.read_text(encoding="utf-8")) \
                        if TRAINING_DATA_PATH.exists() else {"texts": [], "labels": []}
    last_run_ts     = load_last_run()

    print("🚀 Starting market bot...")
    suggest_confidence_threshold(0.75)
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
