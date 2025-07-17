import time
import re
import json
from pathlib import Path
from datetime import datetime, date, timedelta

import requests
import feedparser
import yfinance as yf
from bs4 import BeautifulSoup

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN   = "7623921356:AAGTIO3DP-bdUFj_6ODh4Z2mDLHdHxebw3M"
TELEGRAM_CHAT_ID = "5528794335"

# === RSS & BENZINGA CONFIG ===
RSS_URL         = "https://finance.yahoo.com/rss/topstories"
BENZINGA_API_KEY= "bz.XAO6BCTUMYPFGHXXL7SJ3ZU4IRRTFRE7"
BENZINGA_URL    = "https://api.benzinga.com/api/v2/news"

# === RATE LIMIT & PERSISTENT FILES ===
RATE_LIMIT_SECONDS  = 1800  # 30m per ticker
SENT_LOG_PATH       = Path("sent_titles.txt")
RATE_LIMIT_PATH     = Path("rate_limit.json")
LAST_CAL_RUN_PATH   = Path("last_calendar_run.txt")

sent_news       = set(SENT_LOG_PATH.read_text(encoding="utf-8").splitlines()) \
                    if SENT_LOG_PATH.exists() else set()
rate_limit_data = json.loads(RATE_LIMIT_PATH.read_text(encoding="utf-8")) \
                    if RATE_LIMIT_PATH.exists() else {}
last_cal_run    = LAST_CAL_RUN_PATH.read_text().strip() if LAST_CAL_RUN_PATH.exists() else ""

# === WATCHLIST ===
WATCHLIST = {
    "AAPL": ["AAPL","Apple"],      "MSFT": ["MSFT","Microsoft"],
    "GOOGL":["GOOGL","Google","Alphabet"], "AMZN": ["AMZN","Amazon"],
    "META":["META","Facebook","Meta"], "TSLA":["TSLA","Tesla"],
    "NVDA":["NVDA","NVIDIA"], "AMD":["AMD","Advanced Micro Devices"],
    "INTC":["INTC","Intel"], "NFLX":["NFLX","Netflix"],
    "SPY":["SPY","S&P 500"], "QQQ":["QQQ","Nasdaq"],
    "IWM":["IWM","Russell 2000"], "XOM":["XOM","Exxon","ExxonMobil"],
    "CVX":["CVX","Chevron"], "OXY":["OXY","Occidental"],
    "WMT":["WMT","Walmart"], "COST":["COST","Costco"],
    "TGT":["TGT","Target"], "HD":["HD","Home Depot"],
    "LOW":["LOW","Lowe's"], "JPM":["JPM","JPMorgan"],
    "BAC":["BAC","Bank of America"], "GS":["GS","Goldman Sachs"],
    "MS":["MS","Morgan Stanley"], "WFC":["WFC","Wells Fargo"],
    "BX":["BX","Blackstone"], "UBER":["UBER"], "LYFT":["LYFT"],
    "SNOW":["SNOW","Snowflake"], "PLTR":["PLTR","Palantir"],
    "CRM":["CRM","Salesforce"], "ADBE":["ADBE","Adobe"],
    "SHOP":["SHOP","Shopify"], "PYPL":["PYPL","PayPal"],
    "SQ":["SQ","Block"], "COIN":["COIN","Coinbase"],
    "ROKU":["ROKU"], "BABA":["BABA","Alibaba"], "JD":["JD","JD.com"],
    "NIO":["NIO"], "LI":["LI","Li Auto"], "XPEV":["XPEV","XPeng"],
    "LMT":["LMT","Lockheed Martin"], "NOC":["NOC","Northrop Grumman"],
    "RTX":["RTX","Raytheon"], "BA":["BA","Boeing"], "GE":["GE","General Electric"],
    "CAT":["CAT","Caterpillar"], "DE":["DE","John Deere"],
    "F":["F","Ford"], "GM":["GM","General Motors"],
    "RIVN":["RIVN","Rivian"], "LCID":["LCID","Lucid"],
    "PFE":["PFE","Pfizer"], "MRNA":["MRNA","Moderna"],
    "JNJ":["JNJ","Johnson & Johnson"], "BMY":["BMY","Bristol Myers"],
    "UNH":["UNH","UnitedHealth"], "MDT":["MDT","Medtronic"],
    "ABBV":["ABBV","AbbVie"], "TMO":["TMO","Thermo Fisher"],
    "SHEL":["SHEL","Shell"], "BP":["BP","British Petroleum"],
    "UL":["UL","Unilever"], "BTI":["BTI","British American Tobacco"],
    "SAN":["SAN","Santander"], "DB":["DB","Deutsche Bank"],
    "VTOL":["VTOL","Bristow Group"], "EVTL":["EVTL","Vertical Aerospace"],
    "EH":["EH","EHang"], "PL":["PL","Planet Labs"], "TT":["TT","Trane"],
    "JCI":["JCI","Johnson Controls"], "RDW":["RDW","Redwire"],
    "LOAR":["LOAR","Loar Holdings"], "PANW":["PANW","Palo Alto Networks"],
    "CRWD":["CRWD","CrowdStrike"], "NET":["NET","Cloudflare"],
    "ZS":["ZS","Zscaler"], "TSM":["TSM","Taiwan Semiconductor"],
    "AVGO":["AVGO","Broadcom"], "MU":["MU","Micron"],
    "TXN":["TXN","Texas Instruments"], "QCOM":["QCOM","Qualcomm"],
    "NKE":["NKE","Nike"],  # added
}

# === TICKER DETECTION ===
TICKER_REGEX = re.compile(r'\$([A-Z]{1,5})\b|\(([A-Z]{1,5})\)')
def match_watchlist_alias(text: str):
    lo = text.lower()
    for t, kws in WATCHLIST.items():
        for kw in kws:
            if kw.lower() in lo:
                return t
    return None

def find_ticker_in_text(text: str):
    up = text.upper()
    # $SYM
    m = TICKER_REGEX.search(text)
    if m:
        return m.group(1) or m.group(2)
    # bare ALL‐CAPS
    for w in up.split():
        if re.fullmatch(r'[A-Z]{1,5}', w):
            return w
    return None

# === TELEGRAM ===
def send_to_telegram(message: str):
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode":"Markdown"
    }
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json=payload, timeout=10
        )
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# === RATE LIMIT ===
def is_rate_limited(ticker: str):
    last = rate_limit_data.get(ticker, 0)
    return (time.time() - last) < RATE_LIMIT_SECONDS

def update_rate_limit(ticker: str):
    rate_limit_data[ticker] = time.time()
    RATE_LIMIT_PATH.write_text(json.dumps(rate_limit_data), encoding="utf-8")

# === ARTICLE FETCH ===
def fetch_article_content(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return " ".join(p.get_text() for p in soup.find_all("p"))
    except Exception as e:
        print(f"❌ Article fetch error: {e}")
        return ""

# === PROCESS & SEND ===
def _should_send(title: str, ticker: str):
    if title in sent_news: return False
    if ticker and is_rate_limited(ticker): return False
    return True

def _alert(title: str, ticker: str, source: str):
    msg = f"🗞 *{source} Alert*\n*{ticker or 'GENERAL'}* — {title}"
    send_to_telegram(msg)
    sent_news.add(title)
    SENT_LOG_PATH.write_text("\n".join(sent_news), encoding="utf-8")
    if ticker:
        update_rate_limit(ticker)

def process_yahoo_entry(entry):
    title = entry.get("title","").strip()
    url   = entry.get("link","")
    ticker = match_watchlist_alias(title) \
             or find_ticker_in_text(title) or "GENERAL"
    if _should_send(title, ticker):
        _alert(title, ticker, "Yahoo")

def analyze_yahoo():
    print("📡 Scanning Yahoo RSS...")
    feed = feedparser.parse(RSS_URL)
    for e in feed.entries:
        process_yahoo_entry(e)
    print("✅ Yahoo done.")

def fetch_benzinga(chunk):
    try:
        r = requests.get(
            BENZINGA_URL,
            params={"tickers":",".join(chunk),"items":50,"token":BENZINGA_API_KEY},
            timeout=10
        )
        r.raise_for_status()
        return r.json().get("news",[])
    except Exception as e:
        print(f"❌ Benzinga error: {e}")
        return []

def process_benzinga_article(a):
    title = a.get("title","").strip()
    url   = a.get("url") or a.get("sourceUrl","")
    ticker= match_watchlist_alias(title) \
            or find_ticker_in_text(title) or "GENERAL"
    if _should_send(title, ticker):
        _alert(title, ticker, "Benzinga")

def analyze_benzinga():
    print("📡 Scanning Benzinga...")
    syms = list(WATCHLIST.keys())
    for i in range(0, len(syms), 20):
        for art in fetch_benzinga(syms[i:i+20]):
            process_benzinga_article(art)
    print("✅ Benzinga done.")

# === CATALYST CALENDAR (today only) ===
def build_catalyst_calendar_text():
    today = date.today()
    lines = [f"*Catalyst Calendar for {today.isoformat()}*"]
    events = []

    for tkr in WATCHLIST:
        tk = yf.Ticker(tkr)
        # earnings
        try:
            ed = tk.calendar.loc["Earnings Date"].iat[0]
            dt = pd.to_datetime(ed).date()
            if dt == today:
                events.append((tkr, "Earnings"))
        except:
            pass
        # dividend
        for d in tk.dividends.index:
            if d.date()==today:
                events.append((tkr, "Dividend"))
        # splits
        for d in tk.splits.index:
            if d.date()==today:
                events.append((tkr, "Split"))

    if not events:
        lines.append("_No catalysts scheduled today_")
    else:
        for tkr, kind in sorted(events):
            lines.append(f"• `{tkr}`  {kind}")

    send_to_telegram("\n".join(lines))

# === MAIN LOOP ===
if __name__=="__main__":
    print("🚀 Starting market bot...")
    while True:
        try:
            analyze_yahoo()
            analyze_benzinga()

            # daily catalyst text once
            today_str = date.today().isoformat()
            if today_str != last_cal_run:
                build_catalyst_calendar_text()
                LAST_CAL_RUN_PATH.write_text(today_str, encoding="utf-8")
                last_cal_run = today_str

            print("⏲ Sleeping 60s\n")
            time.sleep(60)
        except Exception as e:
            print(f"💥 Main loop error: {e}")
            time.sleep(10)
