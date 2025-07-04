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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

import nltk
from nltk.corpus import wordnet

# ─── Ensure WordNet ───────────────────────────────────────────────────────────
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# ─── CONFIG & PATHS ────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5528794335")
BENZINGA_API_KEY = os.getenv("BENZINGA_API_KEY", "bz.XAO6BCTUMYPFGHXXL7SJ3ZU4IRRTFRE7")

RSS_URL              = "https://finance.yahoo.com/rss/topstories"
BENZINGA_URL         = "https://api.benzinga.com/api/v2/news"
CONFIDENCE_THRESHOLD = 60
NET_THRESHOLD        = 0.2
RATE_LIMIT_SECONDS   = 1800  # 30m per ticker

SENT_LOG_PATH       = Path("sent_titles.txt")
RATE_LIMIT_LOG_PATH = Path("rate_limit.json")
TRAINING_DATA_PATH  = Path("training_data.json")
TICKER_DATA_PATH    = Path("ticker_training.json")
LAST_RUN_PATH       = Path("last_run.json")
ALL_TICKERS_PATH    = Path("all_tickers.json")

sent_news       = set()
rate_limit_data = {}
training_data   = {"texts": [], "labels": []}
last_run_ts     = 0.0

# ─── LOAD “ALL TICKERS” UNIVERSE ───────────────────────────────────────────────
ALL_TICKERS = json.loads(ALL_TICKERS_PATH.read_text()) if ALL_TICKERS_PATH.exists() else []
TICKER_RE   = re.compile(r"\b(" + "|".join(re.escape(s) for s in ALL_TICKERS) + r")\b", re.IGNORECASE)

# ─── YOUR WATCHLIST (full) ─────────────────────────────────────────────────────
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
    "UBER": ["UBER", "Uber"],
    "LYFT": ["LYFT", "Lyft"],
    "SNOW": ["SNOW", "Snowflake"],
    "PLTR": ["PLTR", "Palantir"],
    "CRM": ["CRM", "Salesforce"],
    "ADBE": ["ADBE", "Adobe"],
    "SHOP": ["SHOP", "Shopify"],
    "PYPL": ["PYPL", "PayPal"],
    "SQ": ["SQ", "Block"],
    "COIN": ["COIN", "Coinbase"],
    "ROKU": ["ROKU", "Roku"],
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
    "NKE": ["NKE", "Nike"]
}

# ─── OVERRIDES & KEYWORD LISTS ────────────────────────────────────────────────
BULLISH_OVERRIDES = [
    "dividend","buyback","upgrade","beat estimates",
    "surge","outperform","raise","equity award","award",
    # ◀─ newly added:
    "rise","rises","jump","jumps","gain","gains"
]
BEARISH_OVERRIDES = [
    "downgrade","miss estimates","warning","cut",
    "plunge","crash","bankruptcy"
]

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

# ─── MODELS ───────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
model      = LogisticRegression()
model_lock = threading.Lock()

ticker_vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
ticker_model      = OneVsRestClassifier(LogisticRegression())

finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model     = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def train_model():
    texts  = training_data.get("texts", [])
    labels = training_data.get("labels", [])
    if len(texts) < 10:
        print("⚠️ Not enough sentiment data")
        return
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

def classify_text(text: str):
    if not training_data.get("texts"):
        return None, None
    X = vectorizer.transform([text])
    p = model.predict(X)[0]
    c = model.predict_proba(X)[0].max() * 100
    return p, round(c,2)

def train_ticker_model():
    if not TICKER_DATA_PATH.exists(): return
    d = json.loads(TICKER_DATA_PATH.read_text())
    X = ticker_vectorizer.fit_transform(d["headlines"])
    y = d["tickers"]
    ticker_model.fit(X,y)
    print(f"✅ Trained ticker model on {len(y)} samples")

def ml_predict_ticker(headline: str) -> str|None:
    if not TICKER_DATA_PATH.exists(): return None
    Xp = ticker_vectorizer.transform([headline])
    return ticker_model.predict(Xp)[0]

def record_feedback(title: str, label: str):
    training_data.setdefault("texts",[]).append(title)
    training_data.setdefault("labels",[]).append(label)
    TRAINING_DATA_PATH.write_text(json.dumps(training_data), encoding="utf-8")

def record_ticker_feedback(headline: str, correct: str):
    data = json.loads(TICKER_DATA_PATH.read_text()) if TICKER_DATA_PATH.exists() else {"headlines":[],"tickers":[]}
    data["headlines"].append(headline)
    data["tickers"].append(correct)
    TICKER_DATA_PATH.write_text(json.dumps(data), encoding="utf-8")

# ─── SENTIMENT HELPERS ─────────────────────────────────────────────────────────
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

# ─── VIX GAUGE ─────────────────────────────────────────────────────────────────
def get_vix_level():
    try:
        h = yf.Ticker("^VIX").history(period="1d",interval="1m")
        if h.empty:
            return None,"⚠️ VIX data unavailable"
        val = h["Close"].iloc[-1]
        if val<14:    lbl="🟢 Low Fear"
        elif val<20:  lbl="�acağız Normal"
        elif val<25:  lbl="🟠 Caution"
        elif val<30:  lbl="🔴 High Fear"
        else:         lbl="🚨 Panic"
        return round(val,2),lbl
    except Exception as e:
        return None,f"❌ VIX error: {e}"

# ─── TELEGRAM & RATE LIMIT ─────────────────────────────────────────────────────
def send_to_telegram(msg: str, return_message_id: bool=False):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id":TELEGRAM_CHAT_ID, "text":msg, "parse_mode":"Markdown"}
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    print(f"✅ Telegram {r.status_code}")
    if return_message_id:
        return r.json()["result"]["message_id"]
    return None

def is_rate_limited(ticker: str) -> bool:
    last = rate_limit_data.get(ticker,0)
    return (time.time() - last) < RATE_LIMIT_SECONDS

def update_rate_limit(ticker: str):
    rate_limit_data[ticker] = time.time()
    RATE_LIMIT_LOG_PATH.write_text(json.dumps(rate_limit_data), encoding="utf-8")

# ─── TICKER FINDERS ────────────────────────────────────────────────────────────
_ticker_cache={}
def is_valid_ticker(sym: str)->bool:
    s=sym.upper()
    if s in _ticker_cache:
        return _ticker_cache[s]
    try:
        info=yf.Ticker(s).info
        valid = info.get("regularMarketPrice") is not None
    except:
        valid=False
    _ticker_cache[s]=valid
    return valid

def match_watchlist_alias(text: str)->str|None:
    lo=text.lower()
    for sym,alts in WATCHLIST.items():
        for a in alts:
            if a.lower() in lo:
                return sym
    return None

def find_ticker_in_text(text: str)->str|None:
    """
    Only match explicit tickers:
      1) $SYM
      2) (SYM)
    """
    up=text.upper()
    # 1) $SYM
    m=re.search(r"\$([A-Z]{1,5})\b",up)
    if m and is_valid_ticker(m.group(1)):
        return m.group(1)
    # 2) (SYM)
    m=re.search(r"\(([A-Z]{1,5})\)",up)
    if m and is_valid_ticker(m.group(1)):
        return m.group(1)
    return None

# ─── CONFIDENCE ───────────────────────────────────────────────────────────────
def calculate_confidence(title:str)->(int,str):
    hl=title.lower()
    hi=sum(w in hl for w in CRITICAL_KEYWORDS)
    st=sum(w in hl for w in STRONG_KEYWORDS)
    md=sum(w in hl for w in MODERATE_KEYWORDS)
    pr=sum(w in hl for w in PRIORITY_KEYWORDS)
    sc=min(100,hi*30+st*20+md*10+pr*5)
    lbl="High" if sc>=80 else "Medium" if sc>=CONFIDENCE_THRESHOLD else "Low"
    return sc,lbl

# ─── PERSISTENCE ───────────────────────────────────────────────────────────────
def load_last_run()->float:
    return float(LAST_RUN_PATH.read_text()) if LAST_RUN_PATH.exists() else 0.0
def save_last_run(ts:float):
    LAST_RUN_PATH.write_text(str(ts), encoding="utf-8")

# ─── HYBRID WORKFLOW ───────────────────────────────────────────────────────────
def summarize_text(text:str, max_paras:int=5)->str:
    paras=text.split("\n")
    return "\n".join(paras[:max_paras])

def validate_full_article(url, title, ticker, msg_id, head_lbl, head_conf, head_sent):
    full     = fetch_article_content(url)
    summary  = summarize_text(full, max_paras=5)
    full_sent = analyze_sentiment_net(summary)
    full_lbl  = get_sentiment_label(summary)
    full_conf,_=calculate_confidence(summary)
    if full_lbl!=head_lbl or abs(full_sent-head_sent)>0.2:
        follow=(
            f"🔄 *Updated Sentiment* for `{ticker}`\n"
            f"*Headline:* {head_lbl} (`{head_sent:.2f}`)\n"
            f"*Full-Text:* {full_lbl} (`{full_sent:.2f}`)\n"
            f"🎯 *Confidence:* {full_conf}%"
        )
        send_to_telegram(follow)

# ─── PROCESS ENTRIES ──────────────────────────────────────────────────────────
def process_yahoo_entry(entry):
    global last_run_ts
    if hasattr(entry,"published_parsed"):
        ts=time.mktime(entry.published_parsed)
        if ts<=last_run_ts: return
        last_run_ts=max(last_run_ts,ts)

    title=entry.get("title","").strip()
    url  =entry.get("link","")
    ticker=(
        match_watchlist_alias(title)
        or find_ticker_in_text(title)
        or ml_predict_ticker(title)
        or "GENERAL"
    )
    head_sent    = analyze_sentiment_net(title)
    head_lbl     = get_sentiment_label(title)
    head_conf, head_conf_lbl = calculate_confidence(title)

    if head_conf>=0.75 or head_lbl=="Neutral" or ticker!="GENERAL":
        msg_id=send_to_telegram(
            f"🗞 *Yahoo Alert*\n*{ticker}* — {title}\n"
            f"📈 Sentiment: *{head_lbl}* (`{head_sent:.2f}`)\n"
            f"🎯 Confidence: *{head_conf}%* ({head_conf_lbl})",
            return_message_id=True
        )
        if url:
            threading.Thread(
                target=validate_full_article,
                args=(url,title,ticker,msg_id,head_lbl,head_conf,head_sent),
                daemon=True
            ).start()

def process_benzinga_article(a):
    title=a.get("title","").strip()
    url  =a.get("url") or a.get("sourceUrl","")
    ticker=(
        match_watchlist_alias(title)
        or find_ticker_in_text(title)
        or ml_predict_ticker(title)
        or "GENERAL"
    )
    head_sent    = analyze_sentiment_net(title)
    head_lbl     = get_sentiment_label(title)
    head_conf, head_conf_lbl = calculate_confidence(title)

    if head_conf>=0.75 or head_lbl=="Neutral" or ticker!="GENERAL":
        msg_id=send_to_telegram(
            f"🗞 *Benzinga Alert*\n*{ticker}* — {title}\n"
            f"📈 Sentiment: *{head_lbl}* (`{head_sent:.2f}`)\n"
            f"🎯 Confidence: *{head_conf}%* ({head_conf_lbl})",
            return_message_id=True
        )
        if url:
            threading.Thread(
                target=validate_full_article,
                args=(url,title,ticker,msg_id,head_lbl,head_conf,head_sent),
                daemon=True
            ).start()

# ─── SCANNERS ─────────────────────────────────────────────────────────────────
def analyze_yahoo():
    global last_run_ts
    last_run_ts=load_last_run()
    feed=feedparser.parse(RSS_URL)
    for e in feed.entries:
        process_yahoo_entry(e)
    save_last_run(last_run_ts)

def fetch_benzinga(chunk):
    try:
        r=requests.get(
            BENZINGA_URL,
            params={"tickers":",".join(chunk),"items":50,"token":BENZINGA_API_KEY},
            timeout=10
        )
        r.raise_for_status()
        return r.json().get("news",[])
    except:
        return []

def analyze_benzinga():
    syms=list(WATCHLIST.keys())
    for i in range(0,len(syms),20):
        for art in fetch_benzinga(syms[i:i+20]):
            process_benzinga_article(art)

def fetch_article_content(url:str)->str:
    try:
        r=requests.get(url,timeout=10)
        soup=BeautifulSoup(r.text,"html.parser")
        return " ".join(p.get_text() for p in soup.find_all("p"))
    except:
        return ""

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
if __name__=="__main__":
    if SENT_LOG_PATH.exists():
        sent_news=set(SENT_LOG_PATH.read_text().splitlines())
    if RATE_LIMIT_LOG_PATH.exists():
        rate_limit_data=json.loads(RATE_LIMIT_LOG_PATH.read_text())
    if TRAINING_DATA_PATH.exists():
        training_data=json.loads(TRAINING_DATA_PATH.read_text())
    last_run_ts=load_last_run()

    train_model()
    train_ticker_model()

    while True:
        analyze_yahoo()
        analyze_benzinga()
        time.sleep(60)
