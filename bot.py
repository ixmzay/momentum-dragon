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

# ─── Ensure WordNet ─────────────────────────────────────────────────
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# ─── CONFIG & PATHS ────────────────────────────────────────────────────
TELEGRAM_TOKEN     = os.getenv("7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M", "")
TELEGRAM_CHAT_ID   = os.getenv("5528794335", "")
BENZINGA_API_KEY   = os.getenv("bz.XAO6BCTUMYPFGHXXL7SJ3ZU4IRRTFRE7", "")

RSS_URL            = "https://finance.yahoo.com/rss/topstories"
BENZINGA_URL       = "https://api.benzinga.com/api/v2/news"

CONFIDENCE_THRESHOLD = 60
NET_THRESHOLD        = 0.2
RATE_LIMIT_SECONDS   = 1800  # seconds per ticker cooldown

SENT_LOG_PATH           = Path("sent_titles.txt")
RATE_LIMIT_LOG_PATH     = Path("rate_limit.json")
TRAINING_DATA_PATH      = Path("training_data.json")
LAST_RUN_YAHOO_PATH     = Path("last_run_yahoo.json")
LAST_RUN_BENZ_PATH      = Path("last_run_benzinga.json")

# ─── GLOBAL STORAGE ─────────────────────────────────────────────────────
sent_news           = set()
rate_limit_data     = {}
training_data       = {"texts": [], "labels": []}
last_run_yahoo_ts   = 0.0
last_run_benz_ts    = 0.0

# ─── WATCHLIST (equities + crypto) ────────────────────────────────────
WATCHLIST = {
    # Equities
    "AAPL": ["AAPL","Apple"],    "MSFT": ["MSFT","Microsoft"],
    "GOOGL": ["GOOGL","Google","Alphabet"],   "AMZN": ["AMZN","Amazon"],
    "META": ["META","Meta","Facebook"],       "TSLA": ["TSLA","Tesla"],
    "NVDA": ["NVDA","NVIDIA"],                "AMD": ["AMD","Advanced Micro Devices"],
    "INTC": ["INTC","Intel"],                 "NFLX": ["NFLX","Netflix"],
    "SPY": ["SPY","S&P 500"],                 "QQQ": ["QQQ","Nasdaq"],
    "IWM": ["IWM","Russell 2000"],            "XOM": ["XOM","Exxon","ExxonMobil"],
    "CVX": ["CVX","Chevron"],                 "OXY": ["OXY","Occidental"],
    "WMT": ["WMT","Walmart"],                 "COST": ["COST","Costco"],
    "TGT": ["TGT","Target"],                  "HD": ["HD","Home Depot"],
    "LOW": ["LOW","Lowe's"],                  "JPM": ["JPM","JPMorgan"],
    "BAC": ["BAC","Bank of America"],         "GS": ["GS","Goldman Sachs"],
    "MS": ["MS","Morgan Stanley"],            "WFC": ["WFC","Wells Fargo"],
    "BX": ["BX","Blackstone"],                "UBER": ["UBER","Uber"],
    "LYFT": ["LYFT","Lyft"],                  "SNOW": ["SNOW","Snowflake"],
    "PLTR": ["PLTR","Palantir"],              "CRM": ["CRM","Salesforce"],
    "ADBE": ["ADBE","Adobe"],                 "SHOP": ["SHOP","Shopify"],
    "PYPL": ["PYPL","PayPal"],                "SQ": ["SQ","Block"],
    "COIN": ["COIN","Coinbase"],              "ROKU": ["ROKU","Roku"],
    "BABA": ["BABA","Alibaba"],               "JD": ["JD","JD.com"],
    "NIO": ["NIO","NIO"],                     "LI": ["LI","Li Auto"],
    "XPEV": ["XPEV","XPeng"],                 "LMT": ["LMT","Lockheed Martin"],
    "NOC": ["NOC","Northrop Grumman"],        "RTX": ["RTX","Raytheon"],
    "BA": ["BA","Boeing"],                    "GE": ["GE","General Electric"],
    "CAT": ["CAT","Caterpillar"],             "DE": ["DE","John Deere"],
    "F": ["F","Ford"],                        "GM": ["GM","General Motors"],
    "RIVN": ["RIVN","Rivian"],                "LCID": ["LCID","Lucid"],
    "PFE": ["PFE","Pfizer"],                  "MRNA": ["MRNA","Moderna"],
    "JNJ": ["JNJ","Johnson & Johnson"],       "BMY": ["BMY","Bristol Myers"],
    "UNH": ["UNH","UnitedHealth"],            "MDT": ["MDT","Medtronic"],
    "ABBV": ["ABBV","AbbVie"],                "TMO": ["TMO","Thermo Fisher"],
    "SHEL": ["SHEL","Shell"],                 "BP": ["BP","British Petroleum"],
    "UL": ["UL","Unilever"],                  "BTI": ["BTI","British American Tobacco"],
    "SAN": ["SAN","Santander"],               "DB": ["DB","Deutsche Bank"],
    "VTOL": ["VTOL","Bristow Group"],         "EVTL": ["EVTL","Vertical Aerospace"],
    "EH": ["EH","EHang"],                     "PL": ["PL","Planet Labs"],
    "TT": ["TT","Trane"],                     "JCI": ["JCI","Johnson Controls"],
    "RDW": ["RDW","Redwire"],                 "LOAR": ["LOAR","Loar Holdings"],
    "PANW": ["PANW","Palo Alto Networks"],    "CRWD": ["CRWD","CrowdStrike"],
    "NET": ["NET","Cloudflare"],              "ZS": ["ZS","Zscaler"],
    "TSM": ["TSM","Taiwan Semiconductor"],    "AVGO": ["AVGO","Broadcom"],
    "MU": ["MU","Micron"],                    "TXN": ["TXN","Texas Instruments"],
    "QCOM": ["QCOM","Qualcomm"],              "NKE": ["NKE","Nike"],
    "FDP": ["FDP","Del Monte Foods"],

    # Crypto
    "BTC": ["BTC","Bitcoin"], "ETH": ["ETH","Ethereum"],
    "LTC": ["LTC","Litecoin"], "DOGE": ["DOGE","Dogecoin"],
    "ADA": ["ADA","Cardano"],  "XRP": ["XRP","Ripple"],
    "SOL": ["SOL","Solana"],   "DOT": ["DOT","Polkadot"],
    "SHIB": ["SHIB","Shiba Inu"],
}

# ─── OVERRIDES & KEYWORDS ───────────────────────────────────────────────
BULLISH_OVERRIDES = ["dividend","buyback","upgrade","beat estimates","surge","outperform","raise","rise","jump","gain","rally"]
BEARISH_OVERRIDES = ["downgrade","miss estimates","warning","cut","plunge","crash","bankruptcy"]

BASE_CRITICAL   = ["bankruptcy","insider trading","sec investigation","fda approval"]
BASE_STRONG     = ["upgrade","beat estimates","surge","outperform","warning","cut"]
BASE_MODERATE   = ["buy","growth","profit","decline","drop","loss"]
BASE_PRIORITY   = ["earnings","price target","guidance","dividend","buyback","merger","acquisition","ipo","layoff","revenue"]

def expand_synonyms(words):
    syns = set(words)
    for w in words:
        for syn in wordnet.synsets(w):
            for lem in syn.lemmas():
                syns.add(lem.name().lower().replace("_"," "))
    return list(syns)

CRITICAL_KEYWORDS = expand_synonyms(BASE_CRITICAL)
STRONG_KEYWORDS   = expand_synonyms(BASE_STRONG)
MODERATE_KEYWORDS = expand_synonyms(BASE_MODERATE)
PRIORITY_KEYWORDS = expand_synonyms(BASE_PRIORITY)

# ─── ML & FinBERT SETUP ───────────────────────────────────────────────
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
model      = LogisticRegression()
model_lock = threading.Lock()

finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model     = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

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

def analyze_sentiment_probs(text):
    inp = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    out = finbert_model(**inp)
    return F.softmax(out.logits, dim=1).detach().numpy()[0]

def analyze_sentiment_net(text):
    p = analyze_sentiment_probs(text)
    return float(p[2] - p[0])

def analyze_sentiment_argmax(text):
    p = analyze_sentiment_probs(text)
    return ["Bearish","Neutral","Bullish"][int(p.argmax())]

def get_sentiment_label(text):
    txt = text.strip().lower()
    if txt.endswith("?"):
        return "Neutral"
    if any(kw in txt for kw in BULLISH_OVERRIDES):
        return "Bullish"
    if any(kw in txt for kw in BEARISH_OVERRIDES):
        return "Bearish"
    return analyze_sentiment_argmax(text)

# ─── VIX GAUGE ─────────────────────────────────────────────────────
def get_vix_level():
    try:
        hist = yf.Ticker("^VIX").history(period="1d", interval="1m")
        if hist.empty:
            return None, "⚠️ VIX data unavailable"
        v = hist["Close"].iloc[-1]
        lbl = (
            "🟢 Low Fear" if v < 14 else
            "🟡 Normal"   if v < 20 else
            "🟠 Caution"  if v < 25 else
            "🔴 High Fear" if v < 30 else
            "🚨 Panic"
        )
        return round(v, 2), lbl
    except Exception as e:
        return None, f"❌ VIX error: {e}"

# ─── TELEGRAM & RATE LIMIT ─────────────────────────────────────────
def send_to_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"Markdown"},
        timeout=10
    )
    try:
        resp.raise_for_status()
        return resp.json().get("result",{}).get("message_id")
    except:
        print("❌ Telegram error:", resp.text)

def is_rate_limited(ticker):
    last = rate_limit_data.get(ticker, 0)
    return (time.time() - last) < RATE_LIMIT_SECONDS

def update_rate_limit(ticker):
    rate_limit_data[ticker] = time.time()
    RATE_LIMIT_LOG_PATH.write_text(json.dumps(rate_limit_data), encoding="utf-8")

# ─── TICKER MATCHING & VALIDATION ───────────────────────────────
_ticker_cache = {}

def is_valid_ticker(sym):
    s = sym.upper()
    if s in _ticker_cache:
        return _ticker_cache[s]
    try:
        info = yf.Ticker(s).info
        valid = info.get("regularMarketPrice") is not None
    except:
        valid = False
    _ticker_cache[s] = valid
    return valid

def match_watchlist_alias(text):
    lo = text.lower()
    for sym,alts in WATCHLIST.items():
        for a in alts:
            if re.search(rf"\b{re.escape(a.lower())}\b", lo):
                return sym
    return None

def find_ticker_in_text(text):
    up = text.upper()
    m = re.search(r"\$([A-Z]{1,5})\b", up)
    if m and is_valid_ticker(m.group(1)):
        return m.group(1)
    m = re.search(r"\(([A-Z]{1,5})\)", up)
    if m and is_valid_ticker(m.group(1)):
        return m.group(1)
    return None

def calculate_confidence(title):
    hl = title.lower()
    hi = sum(w in hl for w in CRITICAL_KEYWORDS)
    st = sum(w in hl for w in STRONG_KEYWORDS)
    md = sum(w in hl for w in MODERATE_KEYWORDS)
    pr = sum(w in hl for w in PRIORITY_KEYWORDS)
    sc = min(100, hi*30 + st*20 + md*10 + pr*5)
    lbl = "High" if sc >= 80 else "Medium" if sc >= CONFIDENCE_THRESHOLD else "Low"
    return sc, lbl

# ─── PERSISTENCE ──────────────────────────────────────────────────
def load_last_run(path):
    return float(path.read_text()) if path.exists() else 0.0

def save_last_run(path, ts):
    path.write_text(str(ts), encoding="utf-8")

def record_feedback(title,label):
    training_data.setdefault("texts", []).append(title)
    training_data.setdefault("labels", []).append(label)
    TRAINING_DATA_PATH.write_text(json.dumps(training_data), encoding="utf-8")

# ─── ARTICLE SCRAPE & VALIDATION ───────────────────────────────
def fetch_article_content(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return " ".join(p.get_text() for p in soup.find_all("p"))
    except:
        return ""

def validate_full_article(url, ticker, head_lbl, head_val, msg_id):
    full = fetch_article_content(url)
    snippet = "\n".join(full.splitlines()[:5])
    fs = analyze_sentiment_net(snippet)
    flbl = get_sentiment_label(snippet)
    if flbl != head_lbl or abs(fs - head_val) > 0.2:
        follow = (
            f"🔄 *Updated Sentiment for {ticker}*\n"
            f"*Headline:* {head_lbl} (`{head_val:.2f}`)\n"
            f"*Full-Text:* {flbl} (`{fs:.2f}`)\n"
        )
        send_to_telegram(follow)
        record_feedback(url or ticker, flbl)

GENERIC_STARTS = ("how ","what ","why ","when ","where ","who ","analyst report")

def process_entry(title, url, source):
    global last_run_yahoo_ts, last_run_benz_ts

    if title.strip().endswith("?") or title.lower().startswith(GENERIC_STARTS):
        ticker = "GENERAL"
    else:
        ticker = match_watchlist_alias(title) or find_ticker_in_text(title) or "GENERAL"

    head_val, head_lbl = analyze_sentiment_net(title), get_sentiment_label(title)
    head_conf, head_conf_lbl = calculate_confidence(title)

    if head_conf >= CONFIDENCE_THRESHOLD or head_lbl == "Neutral" or ticker != "GENERAL":
        msg = (
            f"🗞 *{source} Alert*\n"
            f"*{ticker}* — {title}\n"
            f"📈 Sentiment: *{head_lbl}* (`{head_val:.2f}`)\n"
            f"🎯 Confidence: *{head_conf}%* ({head_conf_lbl})"
        )
        msg_id = send_to_telegram(msg)
        if url:
            threading.Thread(
                target=validate_full_article,
                args=(url, ticker, head_lbl, head_val, msg_id),
                daemon=True
            ).start()

def process_yahoo_entry(entry):
    global last_run_yahoo_ts
    if hasattr(entry, "published_parsed"):
        ts = time.mktime(entry.published_parsed)
        if ts <= last_run_yahoo_ts:
            return
        last_run_yahoo_ts = max(last_run_yahoo_ts, ts)

    title = entry.get("title", "").strip()
    url   = entry.get("link", "")
    process_entry(title, url, "Yahoo")

def process_benzinga_article(a):
    global last_run_benz_ts
    ts = a.get("date", 0)
    if ts <= last_run_benz_ts:
        return
    last_run_benz_ts = max(last_run_benz_ts, ts)

    title = a.get("title", "").strip()
    url   = a.get("url") or a.get("sourceUrl", "")
    process_entry(title, url, "Benzinga")

def analyze_yahoo():
    global last_run_yahoo_ts
    last_run_yahoo_ts = load_last_run(LAST_RUN_YAHOO_PATH)
    for e in feedparser.parse(RSS_URL).entries:
        process_yahoo_entry(e)
    save_last_run(LAST_RUN_YAHOO_PATH, last_run_yahoo_ts)

def analyze_benzinga():
    global last_run_benz_ts
    last_run_benz_ts = load_last_run(LAST_RUN_BENZ_PATH)
    syms = list(WATCHLIST.keys())
    for i in range(0, len(syms), 20):
        resp = requests.get(
            BENZINGA_URL,
            params={"tickers": ",".join(syms[i:i+20]), "items": 50, "token": BENZINGA_API_KEY},
            timeout=10
        )
        data = resp.json().get("news", [])
        for art in data:
            process_benzinga_article(art)
    save_last_run(LAST_RUN_BENZ_PATH, last_run_benz_ts)

if __name__=="__main__":
    # load persistence
    if SENT_LOG_PATH.exists():
        sent_news = set(SENT_LOG_PATH.read_text().splitlines())
    if RATE_LIMIT_LOG_PATH.exists():
        rate_limit_data = json.loads(RATE_LIMIT_LOG_PATH.read_text())
    if TRAINING_DATA_PATH.exists():
        training_data = json.loads(TRAINING_DATA_PATH.read_text())

    train_model()
    print("🚀 Bot started.")
    while True:
        try:
            analyze_yahoo()
            analyze_benzinga()
            time.sleep(60)
        except Exception as e:
            print("💥 Error:", e)
            time.sleep(10)
