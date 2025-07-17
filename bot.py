#!/usr/bin/env python3
import os
import re
import json
import time
from pathlib import Path
from datetime import datetime, date, time as dt_time, timedelta

import requests
import feedparser
import yfinance as yf
from bs4 import BeautifulSoup
from icalendar import Calendar, Event, Alarm
import pytz
import pandas as pd
import numpy as np

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# === RSS & BENZINGA CONFIG ===
RSS_URL          = "https://finance.yahoo.com/rss/topstories"
BENZINGA_API_KEY = os.environ.get("BENZINGA_API_KEY", "")
BENZINGA_URL     = "https://api.benzinga.com/api/v2/news"

# === LIMITS & PATHS ===
RATE_LIMIT_SECONDS   = 1800  # 30-minute cooldown
SENT_LOG_PATH        = Path("sent_titles.txt")
RATE_LIMIT_LOG_PATH  = Path("rate_limit.json")
CAL_RUN_PATH         = Path("last_calendar_run.txt")
CATALYST_CACHE_PATH  = Path("catalyst_cache.json")

# === GLOBAL STORAGE ===
sent_news       = set()
rate_limit_data = {}

# === FULL WATCHLIST ===
WATCHLIST = {
    "AAPL": ["Apple"], "MSFT": ["Microsoft"], "GOOGL": ["Google","Alphabet"],
    "AMZN": ["Amazon"], "META": ["Facebook","Meta"], "TSLA": ["Tesla"],
    "NVDA": ["NVIDIA"], "AMD": ["Advanced Micro Devices"], "INTC": ["Intel"],
    "NFLX": ["Netflix"], "SPY": ["S&P 500"], "QQQ": ["Nasdaq"],
    "IWM": ["Russell 2000"], "XOM": ["Exxon","ExxonMobil"], "CVX": ["Chevron"],
    "OXY": ["Occidental"], "WMT": ["Walmart"], "COST": ["Costco"],
    "TGT": ["Target"], "HD": ["Home Depot"], "LOW": ["Lowe's"],
    "JPM": ["JPMorgan"], "BAC": ["Bank of America"], "GS": ["Goldman Sachs"],
    "MS": ["Morgan Stanley"], "WFC": ["Wells Fargo"], "BX": ["Blackstone"],
    "UBER": ["Uber"], "LYFT": ["Lyft"], "SNOW": ["Snowflake"],
    "PLTR": ["Palantir"], "CRM": ["Salesforce"], "ADBE": ["Adobe"],
    "SHOP": ["Shopify"], "PYPL": ["PayPal"], "SQ": ["Block"],
    "COIN": ["Coinbase"], "ROKU": ["Roku"], "BABA": ["Alibaba"],
    "JD": ["JD.com"], "NIO": ["NIO"], "LI": ["Li Auto"],
    "XPEV": ["XPeng"], "LMT": ["Lockheed Martin"], "NOC": ["Northrop Grumman"],
    "RTX": ["Raytheon"], "BA": ["Boeing"], "GE": ["General Electric"],
    "CAT": ["Caterpillar"], "DE": ["John Deere"], "F": ["Ford"],
    "GM": ["General Motors"], "RIVN": ["Rivian"], "LCID": ["Lucid"],
    "PFE": ["Pfizer"], "MRNA": ["Moderna"], "JNJ": ["Johnson & Johnson"],
    "BMY": ["Bristol Myers"], "UNH": ["UnitedHealth"], "MDT": ["Medtronic"],
    "ABBV": ["AbbVie"], "TMO": ["Thermo Fisher"], "SHEL": ["Shell"],
    "BP": ["British Petroleum"], "UL": ["Unilever"], "BTI": ["British American Tobacco"],
    "SAN": ["Santander"], "DB": ["Deutsche Bank"], "VTOL": ["Bristow Group"],
    "EVTL": ["Vertical Aerospace"], "EH": ["EHang"], "PL": ["Planet Labs"],
    "TT": ["Trane"], "JCI": ["Johnson Controls"], "RDW": ["Redwire"],
    "LOAR": ["Loar Holdings"], "PANW": ["Palo Alto Networks"], "CRWD": ["CrowdStrike"],
    "NET": ["Cloudflare"], "ZS": ["Zscaler"], "TSM": ["Taiwan Semiconductor"],
    "AVGO": ["Broadcom"], "MU": ["Micron"], "TXN": ["Texas Instruments"],
    "QCOM": ["Qualcomm"], "NKE": ["Nike"]
}

# === TICKER DETECTION ===
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
    m = TICKER_REGEX.search(up)
    if m:
        return m.group(1) or m.group(2)
    for w in re.findall(r'\b[A-Z]{2,5}\b', up):
        if w not in {"NEW","ANALYST","REPORT","NEWS","US","THE","AND"}:
            return w
    return None

def detect_ticker(title: str) -> str:
    return (match_watchlist_alias(title)
            or find_ticker_in_text(title)
            or "GENERAL")

# === RATE LIMITING & DEDUPE ===
def is_rate_limited(ticker: str) -> bool:
    last = rate_limit_data.get(ticker, 0)
    return (time.time() - last) < RATE_LIMIT_SECONDS

def update_rate_limit(ticker: str):
    rate_limit_data[ticker] = time.time()
    RATE_LIMIT_LOG_PATH.write_text(json.dumps(rate_limit_data), encoding="utf-8")

# === TELEGRAM ALERTS ===
def send_to_telegram(message: str):
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json=payload, timeout=5
    )
    r.raise_for_status()

def send_alert(title: str, ticker: str, source: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = (
        f"🗞 *{source} Alert*\n"
        f"*{ticker}* — {title}\n"
        f"🕒 {ts}"
    )
    send_to_telegram(msg)
    sent_news.add(title)
    SENT_LOG_PATH.write_text("\n".join(sent_news), encoding="utf-8")
    update_rate_limit(ticker)

# === PROCESS & ALERT for Yahoo RSS ===
def process_yahoo_entry(entry):
    title = entry.get("title", "").strip()
    if not title or title in sent_news:
        return
    ticker = detect_ticker(title)
    if is_rate_limited(ticker):
        return
    send_alert(title, ticker, "Yahoo")

def analyze_yahoo():
    feed = feedparser.parse(RSS_URL)
    for e in feed.entries:
        process_yahoo_entry(e)

# === PROCESS & ALERT for Benzinga API ===
def fetch_benzinga(chunk):
    try:
        r = requests.get(
            BENZINGA_URL,
            params={"tickers": ",".join(chunk), "items": 50, "token": BENZINGA_API_KEY},
            timeout=10
        )
        r.raise_for_status()
        return r.json().get("news", [])
    except:
        return []

def process_benzinga_article(a):
    title = a.get("title", "").strip()
    if not title or title in sent_news:
        return
    ticker = detect_ticker(title)
    if is_rate_limited(ticker):
        return
    send_alert(title, ticker, "Benzinga")

def analyze_benzinga():
    symbols = list(WATCHLIST.keys())
    for i in range(0, len(symbols), 20):
        for art in fetch_benzinga(symbols[i:i+20]):
            process_benzinga_article(art)

# === OVERSOLD SCREENER ===
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def screen_oversold(tickers: list[str]) -> dict[str, dict]:
    oversold = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="2mo", interval="1d")["Close"]
            if len(hist) < 15:
                continue
            today, yesterday = hist.iloc[-1], hist.iloc[-2]
            pct_change = (today / yesterday - 1) * 100
            rsi = compute_rsi(hist).iloc[-1]
            if rsi < 30 or pct_change < -3.0:
                oversold[t] = {"rsi": round(rsi,1), "change%": round(pct_change,2)}
        except:
            pass
    return oversold

# === CATALYST CALENDAR ===
LOCAL_TZ      = pytz.timezone("America/New_York")
TRADING_OPEN  = dt_time(9, 30)
DAYS_AHEAD    = 30

def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}

def save_json(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")

def send_calendar():
    with open("catalyst_calendar.ics", "rb") as f:
        files = {"document": f}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": "🗓 Catalyst calendar"}
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument",
            data=data, files=files, timeout=10
        )
        r.raise_for_status()

def build_catalyst_calendar():
    today = date.today()
    end   = today + timedelta(days=DAYS_AHEAD)
    cache = load_json(CATALYST_CACHE_PATH)
    cal   = Calendar()
    cal.add("prodid", "-//Catalyst Calendar//")
    cal.add("version", "2.0")

    def add_event(dt: datetime, summary: str, desc: str, uid: str):
        if not (TRADING_OPEN <= dt.time() <= dt_time(16,0)):
            return
        evt = Event()
        evt.add("uid", uid)
        evt.add("dtstamp", datetime.now(tz=LOCAL_TZ))
        evt.add("dtstart", LOCAL_TZ.localize(dt))
        evt.add("dtend",   LOCAL_TZ.localize(dt + timedelta(hours=1)))
        evt.add("summary", summary)
        evt.add("description", desc)
        alarm = Alarm()
        alarm.add("action", "DISPLAY")
        alarm.add("description", summary)
        alarm.add("trigger", timedelta(minutes=-30))
        evt.add_component(alarm)
        cal.add_component(evt)

    # earnings
    for ticker in WATCHLIST:
        key = f"{ticker}-earn"
        rec = cache.get(key, {})
        if rec.get("date") != today.isoformat():
            try:
                raw = yf.Ticker(ticker).calendar.loc["Earnings Date"].iat[0]
                dt  = pd.to_datetime(raw).to_pydatetime()
            except:
                dt = None
            cache[key] = {"date": today.isoformat(), "dt": dt.isoformat() if dt else None}
        else:
            dt = datetime.fromisoformat(rec["dt"]) if rec.get("dt") else None
        if dt and today <= dt.date() <= end:
            add_event(dt, f"Earnings: {ticker}", f"{ticker} reports earnings", uid=key)

    # dividends & splits
    for ticker in WATCHLIST:
        key = f"{ticker}-act"
        rec = cache.get(key, {})
        if rec.get("date") != today.isoformat():
            tkr = yf.Ticker(ticker)
            divs = tkr.dividends.index
            spl  = tkr.splits.index
            acts = {
                "dividends": [d.date().isoformat() for d in divs],
                "splits":     [d.date().isoformat() for d in spl]
            }
            cache[key] = {"date": today.isoformat(), "actions": acts}
        acts = cache[key]["actions"]
        for kind, dates in acts.items():
            for d_str in dates:
                d = date.fromisoformat(d_str)
                if today <= d <= end:
                    dt = datetime.combine(d, TRADING_OPEN)
                    add_event(dt, f"{kind.capitalize()}: {ticker}",
                              f"Upcoming {kind} for {ticker}",
                              uid=f"{ticker}-{kind}-{d_str}")

    save_json(CATALYST_CACHE_PATH, cache)
    with open("catalyst_calendar.ics", "wb") as f:
        f.write(cal.to_ical())
    send_calendar()

# === MAIN LOOP ===
def main():
    global sent_news, rate_limit_data

    sent_news       = set(SENT_LOG_PATH.read_text().splitlines()) if SENT_LOG_PATH.exists() else set()
    rate_limit_data = load_json(RATE_LIMIT_LOG_PATH)
    last_cal_run    = CAL_RUN_PATH.read_text().strip() if CAL_RUN_PATH.exists() else ""

    print("🚀 Starting bot…")
    while True:
        try:
            analyze_yahoo()
            analyze_benzinga()

            # oversold scan in first 15m
            now = datetime.now(LOCAL_TZ).time()
            if TRADING_OPEN <= now <= (dt_time(9,45)):
                hits = screen_oversold(list(WATCHLIST.keys()))
                if hits:
                    lines = ["📉 *Oversold Screener*"]
                    for t, stats in hits.items():
                        lines.append(f"{t}: RSI {stats['rsi']} | Δ {stats['change%']}%")
                    send_to_telegram("\n".join(lines))

            # rebuild & push catalyst calendar once per day
            today_str = date.today().isoformat()
            if today_str != last_cal_run:
                build_catalyst_calendar()
                CAL_RUN_PATH.write_text(today_str)
                last_cal_run = today_str

            time.sleep(60)
        except Exception as e:
            print(f"💥 Main loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
