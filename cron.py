import time, requests

URL = "https://YOUR_RENDER_SERVICE.onrender.com/schedule"  # replace this
while True:
    try:
        print("Pinging schedule...")
        requests.get(URL)
    except Exception as e:
        print("Ping failed:", e)
    time.sleep(60)
