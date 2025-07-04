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
CONFIDENCE_THRESHOLD = 60    # keyword‐based medium cutoff
NET_THRESHOLD        = 0.2   # FinBERT net sentiment threshold for GENERAL
RATE_LIMIT_SECONDS   = 1800  # 30‐minute cooldown per ticker

# === LOG / DATA PATHS ===
SENT_LOG_PATH       = Path("sent_titles.txt")
RATE_LIMIT_LOG_PATH = Path("rate_limit.json")
TRAINING_DATA_PATH  = Path("training_data.json")
TICKER_DATA_PATH    = Path("ticker_training.json")
LAST_RUN_PATH       = Path("last_run.json")
ALL_TICKERS_PATH    = Path("all_tickers.json")

# === GLOBAL STATE ===
sent_news       = set()
rate_limit_data = {}
training_data   = {"texts": [], "labels": []}
last_run_ts     = 0.0

# === LOAD ALL VALID TICKERS ===
ALL_TICKERS = json.loads(ALL_TICKERS_PATH.read_text()) if ALL_TICKERS_PATH.exists() else []
TICKER_RE   = re.compile(r"\b(" + "|".join(re.escape(s) for s in ALL_TICKERS) + r")\b")

# === WATCHLIST (for alias matching) ===
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

# === OVERRIDES & KEYWORD LISTS ===
BULLISH_OVERRIDES = ["dividend","buyback","upgrade","beat estimates","surge","outperform"]
BEARISH_OVERRIDES = ["downgrade","miss estimates","warning","cut","plunge","crash"]

BASE_CRITICAL   = ["bankruptcy","insider trading","sec investigation","fda approval"]
BASE_STRONG     = ["upgrade","beat estimates","surge","outperform","warning","cut"]
BASE_MODERATE   = ["buy","growth","profit","decline","drop","loss"]
BASE_PRIORITY   = ["earnings","price target","guidance","dividend","buyback","merger","acquisition","ipo","layoff","revenue"]

def expand_synonyms(words):
    syns = set(words)
    for w in words:
        for syn in wordnet.synsets(w):
            for lem in syn.lemmas():
                syns.add(lem.name().lower().replace('_',' '))
    return list(syns)

CRITICAL_KEYWORDS = expand_synonyms(BASE_CRITICAL)
STRONG_KEYWORDS   = expand_synonyms(BASE_STRONG)
MODERATE_KEYWORDS = expand_synonyms(BASE_MODERATE)
PRIORITY_KEYWORDS = expand_synonyms(BASE_PRIORITY)

# === SENTIMENT & CONFIDENCE ML SETUP ===
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
model      = LogisticRegression()
model_lock = threading.Lock()

ticker_vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
ticker_model      = OneVsRestClassifier(LogisticRegression())

# FinBERT
finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model     = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def suggest_confidence_threshold(pct=0.75):
    scores = [calculate_confidence(t)[0] for t in training_data.get("texts",[])]
    if not scores:
        print("⚠️ No data for threshold suggestion")
        return
    scores.sort()
    idx = int(len(scores)*pct)
    print(f"💡 {int(pct*100)}th percentile score = {scores[min(idx,len(scores)-1)]}")

def train_model():
    texts  = training_data.get("texts",[])
    labels = training_data.get("labels",[])
    if len(texts)<10:
        print("⚠️ Not enough data to train sentiment model")
        return
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X,labels,test_size=0.2,random_state=42)
    model.fit(X_train,y_train)
    print(classification_report(y_test, model.predict(X_test)))

def classify_text(text: str):
    if not training_data.get("texts"):
        return None,None
    X = vectorizer.transform([text])
    p = model.predict(X)[0]
    c = model.predict_proba(X)[0].max()*100
    return p,round(c,2)

def train_ticker_model():
    if not TICKER_DATA_PATH.exists():
        return
    d = json.loads(TICKER_DATA_PATH.read_text())
    X = ticker_vectorizer.fit_transform(d["headlines"])
    y = d["tickers"]
    ticker_model.fit(X,y)
    print(f"✅ Trained ticker model on {len(y)} samples")

def record_ticker_feedback(headline: str, correct: str):
    data = json.loads(TICKER_DATA_PATH.read_text()) if TICKER_DATA_PATH.exists() else {"headlines":[],"tickers":[]}
    data["headlines"].append(headline)
    data["tickers"].append(correct)
    TICKER_DATA_PATH.write_text(json.dumps(data),encoding="utf-8")

def ml_predict_ticker(headline: str) -> str|None:
    if not TICKER_DATA_PATH.exists():
        return None
    Xp = ticker_vectorizer.transform([headline])
    return ticker_model.predict(Xp)[0]

def analyze_sentiment_probs(text: str):
    inp = finbert_tokenizer(text,return_tensors="pt",truncation=True,max_length=512)
    out = finbert_model(**inp)
    return F.softmax(out.logits,dim=1).detach().numpy()[0]

def analyze_sentiment_net(text: str)->float:
    p = analyze_sentiment_probs(text)
    return float(p[2]-p[0])

def analyze_sentiment_argmax(text: str)->str:
    p = analyze_sentiment_probs(text)
    return ["Bearish","Neutral","Bullish"][int(p.argmax())]

def get_sentiment_label(text: str)->str:
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
        h = yf.Ticker("^VIX").history(period="1d",interval="1m")
        if h.empty:
            return None,"⚠️ VIX data unavailable"
        val = h["Close"].iloc[-1]
        if val<14:    lbl="🟢 Low Fear"
        elif val<20:  lbl="🟡 Normal"
        elif val<25:  lbl="🟠 Caution"
        elif val<30:  lbl="🔴 High Fear"
        else:         lbl="🚨 Panic"
        return round(val,2),lbl
    except Exception as e:
        return None,f"❌ VIX error: {e}"

def send_to_telegram(msg:str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload={"chat_id":TELEGRAM_CHAT_ID,"text":msg,"parse_mode":"Markdown"}
    try:
        r = requests.post(url,json=payload,timeout=10)
        r.raise_for_status()
        print(f"✅ Telegram {r.status_code}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def is_rate_limited(ticker:str)->bool:
    last = rate_limit_data.get(ticker,0)
    return (time.time()-last)<RATE_LIMIT_SECONDS

def update_rate_limit(ticker:str):
    rate_limit_data[ticker]=time.time()
    RATE_LIMIT_LOG_PATH.write_text(json.dumps(rate_limit_data),encoding="utf-8")

_ticker_cache={}
def is_valid_ticker(sym:str)->bool:
    s=sym.upper()
    if s in _ticker_cache:
        return _ticker_cache[s]
    try:
        info=yf.Ticker(s).info
        valid=info.get("regularMarketPrice") is not None
    except:
        valid=False
    _ticker_cache[s]=valid
    return valid

def match_watchlist_alias(text:str)->str|None:
    lo=text.lower()
    for sym,alts in WATCHLIST.items():
        for a in alts:
            if a.lower() in lo:
                return sym
    return None

def find_ticker_in_text(text:str)->str|None:
    up=text.upper()
    m=re.search(r"\$([A-Z]{1,5})\b",up)
    if m and is_valid_ticker(m.group(1)):
        return m.group(1)
    m=re.search(r"\(([A-Z]{1,5})\)",up)
    if m and is_valid_ticker(m.group(1)):
        return m.group(1)
    m=TICKER_RE.search(up)
    if m:
        return m.group(1)
    return None

def calculate_confidence(title:str)->(int,str):
    hl=title.lower()
    hi=sum(w in hl for w in CRITICAL_KEYWORDS)
    st=sum(w in hl for w in STRONG_KEYWORDS)
    md=sum(w in hl for w in MODERATE_KEYWORDS)
    pr=sum(w in hl for w in PRIORITY_KEYWORDS)
    sc=min(100,hi*30+st*20+md*10+pr*5)
    lbl="High" if sc>=80 else "Medium" if sc>=CONFIDENCE_THRESHOLD else "Low"
    return sc,lbl

def load_last_run()->float:
    return float(LAST_RUN_PATH.read_text()) if LAST_RUN_PATH.exists() else 0.0

def save_last_run(ts:float):
    LAST_RUN_PATH.write_text(str(ts),encoding="utf-8")

def record_feedback(title:str,label:str):
    training_data.setdefault("texts",[]).append(title)
    training_data.setdefault("labels",[]).append(label)
    TRAINING_DATA_PATH.write_text(json.dumps(training_data),encoding="utf-8")

def should_send_alert(title:str,ticker:str,conf:int,summary:str)->bool:
    if title in sent_news or is_rate_limited(ticker):
        return False
    if ticker!="GENERAL":
        return True
    net=analyze_sentiment_net(summary)
    return conf>=CONFIDENCE_THRESHOLD or abs(net)>=NET_THRESHOLD

def send_alert(title:str,ticker:str,summary:str,conf:int,conf_lbl:str,source:str):
    net     = analyze_sentiment_net(summary)
    sent_lbl= get_sentiment_label(summary)
    v_val,v_lbl = get_vix_level()
    ml_p,ml_c   = classify_text(title)
    ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    parts=[
        f"🗞 *{source} Alert*",
        f"*{ticker}* — {title}",
        f"📈 Sentiment: *{sent_lbl}* (`{net:.2f}`)",
        f"🎯 Confidence: *{conf}%* ({conf_lbl})"
    ]
    if ml_p:
        parts.append(f"🤖 ML: *{ml_p}* ({ml_c}%)")
    parts.append(f"🌪 VIX: *{v_val}* — {v_lbl}  🕒 {ts}")

    send_to_telegram("\n".join(parts))
    sent_news.add(title)
    SENT_LOG_PATH.write_text("\n".join(sent_news),encoding="utf-8")
    update_rate_limit(ticker)

def fetch_article_content(url:str)->str:
    try:
        r=requests.get(url,timeout=10)
        soup=BeautifulSoup(r.text,"html.parser")
        return " ".join(p.get_text() for p in soup.find_all("p"))
    except Exception as e:
        print(f"❌ Article fetch error: {e}")
        return ""

def process_yahoo_entry(entry):
    global last_run_ts
    if hasattr(entry,"published_parsed"):
        ts=time.mktime(entry.published_parsed)
        if ts<=last_run_ts:
            return
        last_run_ts=max(last_run_ts,ts)

    title=entry.get("title","").strip()
    summary=getattr(entry,"summary_detail",{}).get("value","") or entry.get("summary","")
    text=BeautifulSoup(summary or title,"html.parser").get_text().strip()

    print("▶️ Yahoo headline:",title)
    ticker = (
        match_watchlist_alias(title)
        or find_ticker_in_text(title)
        or ml_predict_ticker(title)
        or "GENERAL"
    )
    print(f"   → ticker: {ticker}")

    conf,conf_lbl=calculate_confidence(title)
    if should_send_alert(title,ticker,conf,text):
        print("   → sending alert")
        send_alert(title,ticker,text,conf,conf_lbl,"Yahoo")
    else:
        print("   → filtered out")

def analyze_yahoo():
    global last_run_ts
    last_run_ts=load_last_run()
    print("📡 Scanning Yahoo RSS...")
    feed=feedparser.parse(RSS_URL)
    for e in feed.entries:
        process_yahoo_entry(e)
    save_last_run(last_run_ts)
    print("✅ Yahoo done.")

def process_benzinga_article(a):
    title=a.get("title","").strip()
    url=a.get("url") or a.get("sourceUrl","")
    text=fetch_article_content(url) if url else title

    print("▶️ Benzinga headline:",title)
    ticker = (
        match_watchlist_alias(title)
        or find_ticker_in_text(title)
        or ml_predict_ticker(title)
        or "GENERAL"
    )
    print(f"   → ticker: {ticker}")

    conf,conf_lbl=calculate_confidence(title)
    if should_send_alert(title,ticker,conf,text):
        print("   → sending alert")
        send_alert(title,ticker,text,conf,conf_lbl,"Benzinga")
    else:
        print("   → filtered out")

def analyze_benzinga():
    print("📡 Scanning Benzinga...")
    syms=list(WATCHLIST.keys())
    for i in range(0,len(syms),20):
        for art in fetch_benzinga(syms[i:i+20]):
            process_benzinga_article(art)
    print("✅ Benzinga done.")

def fetch_benzinga(chunk):
    try:
        r=requests.get(
            BENZINGA_URL,
            params={"tickers":",".join(chunk),"items":50,"token":BENZINGA_API_KEY},
            timeout=10
        )
        r.raise_for_status()
        return r.json().get("news",[])
    except Exception as e:
        print(f"❌ Benzinga error: {e}")
        return []

if __name__=="__main__":
    sent_news       = set(SENT_LOG_PATH.read_text().splitlines()) if SENT_LOG_PATH.exists() else set()
    rate_limit_data = json.loads(RATE_LIMIT_LOG_PATH.read_text()) if RATE_LIMIT_LOG_PATH.exists() else {}
    training_data   = json.loads(TRAINING_DATA_PATH.read_text()) if TRAINING_DATA_PATH.exists() else {"texts":[],"labels":[]}
    last_run_ts     = load_last_run()

    print("🚀 Starting market bot…")
    suggest_confidence_threshold(0.75)
    train_model()
    train_ticker_model()

    while True:
        try:
            analyze_yahoo()
            analyze_benzinga()
            print("⏲ Sleeping 60s…\n")
            time.sleep(60)
        except Exception as e:
            print(f"💥 Main loop error: {e}")
            time.sleep(10)
