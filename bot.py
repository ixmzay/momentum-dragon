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

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5528794335")

# === RSS & BENZINGA CONFIG ===
RSS_URL          = "https://finance.yahoo.com/rss/topstories"
BENZINGA_API_KEY = os.getenv("BENZINGA_API_KEY", "bz.XAO6BCTUMYPFGHXXL7SJ3ZU4IRRTFRE7")
BENZINGA_URL     = "https://api.benzinga.com/api/v2/news"

# === LIMITS & PATHS ===
RATE_LIMIT_SECONDS = 1800  # 30‑minute cooldown
SENT_LOG_PATH       = Path("sent_titles.txt")
RATE_LIMIT_LOG_PATH = Path("rate_limit.json")
CAL_RUN_PATH        = Path("last_calendar_run.txt")
CATALYST_CACHE_PATH = Path("catalyst_cache.json")

# === GLOBAL STORAGE ===
sent_news       = set()
rate_limit_data = {}

# === FULL WATCHLIST ===
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
    # fallback to bare all-caps words
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
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json=payload, timeout=5
        )
        r.raise_for_status()
    except Exception as e:
        print(f"Telegram error: {e}")

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
    print("📡 Scanning Yahoo RSS...")
    feed = feedparser.parse(RSS_URL)
    for e in feed.entries:
        process_yahoo_entry(e)
    print("✅ Yahoo done.")

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
    except Exception as e:
        print(f"Benzinga error: {e}")
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
    print("📡 Scanning Benzinga...")
    syms = list(WATCHLIST.keys())
    for i in range(0, len(syms), 20):
        for art in fetch_benzinga(syms[i:i+20]):
            process_benzinga_article(art)
    print("✅ Benzinga done.")

# === CATALYST CALENDAR ===
LOCAL_TZ      = pytz.timezone("America/New_York")
TRADING_OPEN  = dt_time(9, 30)
TRADING_CLOSE = dt_time(16, 0)
DAYS_AHEAD    = 30

def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}

def save_json(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")

def build_catalyst_calendar():
    today = date.today()
    end   = today + timedelta(days=DAYS_AHEAD)
    cache = load_json(CATALYST_CACHE_PATH)
    cal   = Calendar()
    cal.add("prodid", "-//Catalyst Calendar//")
    cal.add("version", "2.0")

    def add_event(dt, summary, desc, uid, recur=False):
        if not (TRADING_OPEN <= dt.time() <= TRADING_CLOSE):
            return
        evt = Event()
        evt.add("uid", uid)
        evt.add("dtstamp", datetime.now(tz=LOCAL_TZ))
        evt.add("dtstart", LOCAL_TZ.localize(dt))
        evt.add("dtend",   LOCAL_TZ.localize(dt + timedelta(hours=1)))
        evt.add("summary", summary)
        evt.add("description", desc)
        if recur:
            evt.add("rrule", {"freq": "monthly", "interval": 3})
        alarm = Alarm()
        alarm.add("action", "DISPLAY")
        alarm.add("description", summary)
        alarm.add("trigger", timedelta(minutes=-30))
        evt.add_component(alarm)
        cal.add_component(evt)

    # Earnings via yfinance.calendar
    for ticker in WATCHLIST:
        key = f"{ticker}-earn"
        rec = cache.get(key, {})
        if rec.get("date") == today.isoformat() and rec.get("dt"):
            dt = datetime.fromisoformat(rec["dt"])
        else:
            tkr = yf.Ticker(ticker)
            try:
                df = tkr.calendar
                raw = df.loc["Earnings Date"].iat[0]
                dt = pd.to_datetime(raw).to_pydatetime()
            except Exception:
                dt = None
            cache[key] = {"date": today.isoformat(), "dt": dt.isoformat() if dt else None}
        if dt:
            add_event(
                dt,
                f"Earnings: {ticker}",
                f"{ticker} reports earnings",
                uid=f"{ticker}-earn",
                recur=True
            )

    # Dividends & Splits
    for ticker in WATCHLIST:
        key = f"{ticker}-act"
        rec = cache.get(key, {})
        if rec.get("date") == today.isoformat():
            acts = rec["actions"]
        else:
            tkr  = yf.Ticker(ticker)
            acts = {
                "divs": tkr.dividends.to_dict(),
                "spl": tkr.splits.to_dict()
            }
            cache[key] = {"date": today.isoformat(), "actions": acts}
        for kind, hist in acts.items():
            if not hist:
                continue
            last = max(hist.keys())
            nxt  = last + timedelta(days=90)
            if today <= nxt <= end:
                add_event(
                    datetime.combine(nxt, TRADING_OPEN),
                    f"{kind.capitalize()}: {ticker}",
                    f"Upcoming {kind} for {ticker}",
                    uid=f"{ticker}-{kind}"
                )

    save_json(CATALYST_CACHE_PATH, cache)
    with open("catalyst_calendar.ics", "wb") as f:
        f.write(cal.to_ical())
    send_to_telegram("🗓 Catalyst calendar updated: `catalyst_calendar.ics`")

# === MAIN LOOP ===
def main():
    global sent_news, rate_limit_data

    # load persistent data
    sent_news       = set(SENT_LOG_PATH.read_text().splitlines()) if SENT_LOG_PATH.exists() else set()
    rate_limit_data = load_json(RATE_LIMIT_LOG_PATH)
    last_cal_run    = CAL_RUN_PATH.read_text().strip() if CAL_RUN_PATH.exists() else ""

    print("🚀 Starting bot with Catalyst calendar…")
    while True:
        try:
            analyze_yahoo()
            analyze_benzinga()

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
