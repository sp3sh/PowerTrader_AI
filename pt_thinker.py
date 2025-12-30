# ============================================================
# PowerTrader AI – THINKER
# Full Rewrite (Trainer-Compatible, Safe, Deterministic)
# ============================================================

import os
import sys
import time
import json
import math
import base64
import requests
import traceback
import multiprocessing as mp
from datetime import datetime
from kucoin.client import Market
from nacl.signing import SigningKey

# ============================================================
# CONFIG
# ============================================================

TF_CHOICES = ['1hour', '2hour', '4hour', '8hour', '12hour', '1day', '1week']
DISTANCE_PCT = 0.5
MIN_CONFIDENCE = 0.20        # HARD FLOOR
MIN_MATCHES = 3             # avoid single-memory hallucinations
BACKTEST_MODE = os.getenv("POWERTRADER_BACKTEST") == "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MARKET = Market(url="https://api.kucoin.com")

# ============================================================
# ROBINHOOD PRICE FEED (LIVE MODE ONLY)
# ============================================================

ROBINHOOD_BASE_URL = "https://trading.robinhood.com"
_RH = None

class RobinhoodMarketData:
    def __init__(self, api_key, private_b64):
        raw = base64.b64decode(private_b64.strip())
        self.key = api_key.strip()
        self.signer = SigningKey(raw)
        self.s = requests.Session()

    def _headers(self, method, path, ts):
        msg = f"{self.key}{ts}{path}{method}"
        sig = self.signer.sign(msg.encode()).signature
        return {
            "x-api-key": self.key,
            "x-timestamp": str(ts),
            "x-signature": base64.b64encode(sig).decode(),
            "Content-Type": "application/json",
        }

    def ask(self, symbol):
        path = f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}"
        ts = int(time.time())
        r = self.s.get(
            ROBINHOOD_BASE_URL + path,
            headers=self._headers("GET", path, ts),
            timeout=10
        )
        r.raise_for_status()
        return float(r.json()["results"][0]["ask_inclusive_of_buy_spread"])

def robinhood_ask(symbol):
    global _RH
    if _RH is None:
        with open("r_key.txt") as f:
            k = f.read()
        with open("r_secret.txt") as f:
            s = f.read()
        _RH = RobinhoodMarketData(k, s)
    return _RH.ask(symbol)

# ============================================================
# UTIL
# ============================================================

def coin_dir(sym):
    return BASE_DIR if sym == "BTC" else os.path.join(BASE_DIR, sym)

def safe_float(x, d=0.0):
    try:
        return float(x)
    except:
        return d

def log_exc():
    print(traceback.format_exc())

# ============================================================
# TRAINING FILE LOADER
# ============================================================

def load_training(tf):
    try:
        with open(f"neural_perfect_threshold_{tf}.txt") as f:
            threshold = float(f.read())

        with open(f"memories_{tf}.txt") as f:
            memories = f.read().split("~")

        with open(f"memory_weights_{tf}.txt") as f:
            weights = [safe_float(x) for x in f.read().split()]

        with open(f"memory_weights_high_{tf}.txt") as f:
            w_high = [safe_float(x) for x in f.read().split()]

        with open(f"memory_weights_low_{tf}.txt") as f:
            w_low = [safe_float(x) for x in f.read().split()]

        return threshold, memories, weights, w_high, w_low
    except:
        return None

# ============================================================
# CORE SIGNAL ENGINE
# ============================================================

def evaluate_candle(candle, training):
    threshold, memories, weights, w_high, w_low = training

    move_sum = 0.0
    high_sum = 0.0
    low_sum = 0.0
    conf = 0.0
    matches = 0

    for i, m in enumerate(memories):
        try:
            parts = m.split("{}")
            pat = safe_float(parts[0].split()[-1])
            diff = abs(candle - pat)

            if diff <= threshold:
                w = weights[i]
                move_sum += pat * w
                high_sum += safe_float(parts[1]) * w_high[i]
                low_sum += safe_float(parts[2]) * w_low[i]
                conf += abs(w)
                matches += 1
        except:
            continue

    if matches < MIN_MATCHES or conf < MIN_CONFIDENCE:
        return None

    return {
        "move": move_sum / conf,
        "high": high_sum / conf,
        "low": low_sum / conf,
        "confidence": conf,
        "matches": matches
    }

# ============================================================
# BACKTEST HARNESS
# ============================================================

def backtest_coin(sym):
    os.chdir(coin_dir(sym))
    results = []

    for tf in TF_CHOICES:
        training = load_training(tf)
        if not training:
            continue

        klines = MARKET.get_kline(f"{sym}-USDT", tf)
        for k in klines[10:]:
            o = float(k[1])
            c = float(k[2])
            candle = 100 * ((c - o) / o)

            res = evaluate_candle(candle, training)
            if not res:
                continue

            results.append({
                "tf": tf,
                "confidence": res["confidence"],
                "matches": res["matches"],
                "move": res["move"]
            })

    with open("backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)

# ============================================================
# LIVE THINKER PROCESS (ONE PER COIN)
# ============================================================

def thinker_process(sym):
    os.chdir(coin_dir(sym))
    tf_index = 0
    state = {tf: {} for tf in TF_CHOICES}

    while True:
        tf = TF_CHOICES[tf_index]

        try:
            k = MARKET.get_kline(f"{sym}-USDT", tf)
            o = float(k[0][1])
            c = float(k[0][2])
            candle = 100 * ((c - o) / o)
        except:
            time.sleep(1)
            continue

        training = load_training(tf)
        if training:
            res = evaluate_candle(candle, training)
        else:
            res = None

        state[tf] = res or {"status": "INACTIVE"}

        tf_index = (tf_index + 1) % len(TF_CHOICES)

        # full sweep → decision
        if tf_index == 0 and not BACKTEST_MODE:
            price = robinhood_ask(f"{sym}-USD")
            messages = []

            for tf in TF_CHOICES:
                r = state[tf]
                if not r or "move" not in r:
                    messages.append(f"INACTIVE {tf}")
                    continue

                projected = price * (1 + r["move"] / 100)
                low = projected * (1 - DISTANCE_PCT / 100)
                high = projected * (1 + DISTANCE_PCT / 100)

                if price > high:
                    messages.append(f"SHORT {tf}")
                elif price < low:
                    messages.append(f"LONG {tf}")
                else:
                    messages.append(f"WITHIN {tf}")

            with open("thinker_output.txt", "w") as f:
                f.write("\n".join(messages))

        time.sleep(0.15)

# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    coins = []
    settings = os.path.join(BASE_DIR, "gui_settings.json")
    if os.path.isfile(settings):
        with open(settings) as f:
            coins = json.load(f).get("coins", [])
    if not coins:
        coins = ["BTC", "ETH", "XRP", "BNB", "DOGE"]

    for c in coins:
        os.makedirs(coin_dir(c), exist_ok=True)

    if BACKTEST_MODE:
        for c in coins:
            backtest_coin(c)
        sys.exit(0)

    procs = []
    for c in coins:
        p = mp.Process(target=thinker_process, args=(c,), daemon=True)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
