import time
import re
import json
from pathlib import Path
from datetime import datetime, date

import requests
import feedparser
import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
from bs4 import BeautifulSoup

# === CONFIG ===
TELEGRAM_TOKEN   = "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M"
TELEGRAM_CHAT_ID = "5528794335"

RSS_URL          = "https://finance.yahoo.com/rss/topstories"
BENZINGA_API_KEY = "bz.XAO6BCTUMYPFGHXXL7SJ3ZU4IRRTFRE7"
BENZINGA_URL     = "https://api.benzinga.com/api/v2/news"

RATE_LIMIT_SECONDS   = 1800  # 30‑minute cooldown per ticker
SENT_LOG_PATH        = Path("sent_titles.txt")
RATE_LIMIT_LOG_PATH  = Path("rate_limit.json")
TICKER_LIST_PATH     = Path("ticker list.txt")

# === PERSISTENT STORAGE (loaded at runtime) ===
sent_news       = set()
rate_limit_data = {}
MASTER_TICKERS  = set()

# === LOAD MASTER TICKERS ===
def load_master_tickers():
    global MASTER_TICKERS
    if TICKER_LIST_PATH.exists():
        lines = [l.strip().upper() for l in TICKER_LIST_PATH.read_text().splitlines() if l.strip()]
        MASTER_TICKERS = set(lines)
    else:
        MASTER_TICKERS = set()  # fallback empty

# === TELEGRAM SENDER ===
def send_to_telegram(message: str):
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json=payload, timeout=5
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# === RATE LIMIT ===
def is_rate_limited(ticker: str) -> bool:
    last = rate_limit_data.get(ticker, 0)
    return (time.time() - last) < RATE_LIMIT_SECONDS

def update_rate_limit(ticker: str):
    rate_limit_data[ticker] = time.time()
    RATE_LIMIT_LOG_PATH.write_text(json.dumps(rate_limit_data), encoding="utf-8")

# === TICKER FINDER ===
TICKER_REGEX = re.compile(r'\$([A-Z]{1,5})\b|\(([A-Z]{1,5})\)')

def find_ticker(text: str) -> str:
    up = text.upper()
    # $SYM or (SYM)
    m = TICKER_REGEX.search(up)
    if m:
        sym = m.group(1) or m.group(2)
        if sym in MASTER_TICKERS:
            return sym
    # bare uppercase words
    for w in up.split():
        if w in MASTER_TICKERS:
            return w
    return "GENERAL"

# === VIX GAUGE ===
def get_vix_level():
    try:
        hist = yf.Ticker("^VIX").history(period="1d", interval="1m")
        latest = hist["Close"].iloc[-1]
        if latest < 14:     lbl = "🟢 Low Fear"
        elif latest < 20:   lbl = "🟡 Normal"
        elif latest < 25:   lbl = "🟠 Caution"
        elif latest < 30:   lbl = "🔴 High Fear"
        else:               lbl = "🚨 Panic"
        return round(latest,2), lbl
    except Exception as e:
        return None, f"❌ VIX error: {e}"

# === ALERT SENDING ===
def send_alert(title: str, ticker: str, source: str):
    vix_val, vix_lbl = get_vix_level()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        f"🗞 *{source} Alert*",
        f"*{ticker}* — {title}",
        f"🌪 VIX: *{vix_val}* — {vix_lbl}",
        f"🕒 {ts}"
    ]
    send_to_telegram("\n".join(lines))
    sent_news.add(title)
    SENT_LOG_PATH.write_text("\n".join(sent_news), encoding="utf-8")
    update_rate_limit(ticker)

def should_send_alert(title: str, ticker: str) -> bool:
    if title in sent_news or is_rate_limited(ticker):
        return False
    return True  # send everything now

# === DAILY CATALYST CALENDAR ===
def send_catalyst_calendar():
    today = date.today().strftime("%Y-%m-%d")
    yec = YahooEarningsCalendar()
    try:
        events = yec.get_earnings_of(today)
    except Exception as e:
        events = []
        print(f"❌ Catalyst fetch error: {e}")

    if not events:
        msg = f"📅 Catalyst Calendar for {today}:\n_No earnings scheduled._"
    else:
        lines = [f"📅 Catalyst Calendar for {today}:"]
        for ev in events:
            sym = ev.get("ticker")
            dt  = ev.get("startdatetime", ev.get("startdatetimeUTC", "Time N/A"))
            lines.append(f"- {sym}: Earnings at {dt}")
        msg = "\n".join(lines)

    send_to_telegram(msg)

# === ARTICLE FETCHER ===
def fetch_article_content(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return " ".join(p.get_text() for p in soup.find_all("p"))
    except Exception as e:
        print(f"❌ Article fetch error: {e}")
        return ""

# === RSS PROCESSING ===
def process_yahoo_entry(entry):
    title = entry.get("title", "").strip()
    link  = entry.get("link", "")
    ticker = find_ticker(title)
    print(f"▶️ Yahoo headline: {title}\n   → ticker: {ticker}")

    if should_send_alert(title, ticker):
        send_alert(title, ticker, "Yahoo")

def analyze_yahoo():
    print("📡 Scanning Yahoo RSS...")
    feed = feedparser.parse(RSS_URL)
    for entry in feed.entries:
        process_yahoo_entry(entry)
    print("✅ Yahoo done.")

# === BENZINGA PROCESSING ===
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

def process_benzinga_article(article):
    title = article.get("title", "").strip()
    ticker = find_ticker(title)
    print(f"▶️ Benzinga headline: {title}\n   → ticker: {ticker}")
    if should_send_alert(title, ticker):
        send_alert(title, ticker, "Benzinga")

def analyze_benzinga():
    print("📡 Scanning Benzinga...")
    # break into chunks of 20 for the API
    symbols = list(MASTER_TICKERS)[:100]  # limit to first 100 for demo; adjust as needed
    for i in range(0, len(symbols), 20):
        for art in fetch_benzinga(symbols[i:i+20]):
            process_benzinga_article(art)
    print("✅ Benzinga done.")

# === MAIN ===
if __name__ == "__main__":
    # load persistence
    if SENT_LOG_PATH.exists():
        sent_news.update(SENT_LOG_PATH.read_text(encoding="utf-8").splitlines())
    if RATE_LIMIT_LOG_PATH.exists():
        rate_limit_data.update(json.loads(RATE_LIMIT_LOG_PATH.read_text(encoding="utf-8")))

    load_master_tickers()
    print("🚀 Starting market bot with Catalyst calendar…")

    # send today's catalysts once at start
    send_catalyst_calendar()

    # main loop
    while True:
        try:
            analyze_yahoo()
            analyze_benzinga()
            time.sleep(60)
        except Exception as e:
            print(f"💥 Main loop error: {e}")
            time.sleep(10)
