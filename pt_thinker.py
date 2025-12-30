# ============================================================
# PowerTrader AI – THINKER (Extended)
# Confidence Scaling · TF Decay · PnL Backtest · Version Sync
# ============================================================

import os
import sys
import time
import json
import math
import base64
import requests
import multiprocessing as mp
from collections import deque
from kucoin.client import Market
from nacl.signing import SigningKey

# ============================================================
# GLOBAL VERSION CONTRACT
# ============================================================

THINKER_VERSION = "2.1.0"
REQUIRED_TRAINER_VERSION = "2.1.0"

# ============================================================
# CONFIG
# ============================================================

TF_CHOICES = ['1hour', '2hour', '4hour', '8hour', '12hour', '1day', '1week']
MIN_CONFIDENCE = 0.25
MIN_MATCHES = 3
DECAY_LIMIT = 5
RECOVERY_HITS = 3
BACKTEST_MODE = os.getenv("POWERTRADER_BACKTEST") == "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MARKET = Market(url="https://api.kucoin.com")

# ============================================================
# CONFIDENCE → SIZE MAP
# ============================================================

def size_multiplier(conf):
    if conf < 0.25:
        return 0.0
    if conf < 0.50:
        return 0.25
    if conf < 0.75:
        return 0.50
    if conf < 1.25:
        return 0.75
    return 1.00

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

# ============================================================
# TRAINING LOADER + HANDSHAKE
# ============================================================

def load_training(tf):
    try:
        with open(f"trainer_version.txt") as f:
            tv = f.read().strip()
        if tv != REQUIRED_TRAINER_VERSION:
            raise RuntimeError("Trainer version mismatch")

        with open(f"memories_{tf}.txt") as f:
            memories = f.read().split("~")

        with open(f"memory_weights_{tf}.txt") as f:
            weights = [safe_float(x) for x in f.read().split()]

        if len(memories) != len(weights):
            raise RuntimeError("Memory / weight mismatch")

        return memories, weights
    except:
        return None

# ============================================================
# CORE EVALUATION
# ============================================================

def evaluate(candle, training):
    memories, weights = training
    score = 0.0
    conf = 0.0
    hits = 0

    for i, m in enumerate(memories):
        try:
            pat, move = m.split("{}")
            pat = safe_float(pat)
            move = safe_float(move)
            diff = abs(candle - pat)

            if diff < 0.5:
                w = weights[i]
                score += move * w
                conf += abs(w)
                hits += 1
        except:
            continue

    if hits < MIN_MATCHES or conf < MIN_CONFIDENCE:
        return None

    return score / conf, conf

# ============================================================
# THINKER PROCESS (PER COIN)
# ============================================================

def thinker(sym):
    os.chdir(coin_dir(sym))

    tf_state = {
        tf: {
            "decay": 0,
            "cooldown": False,
            "hits": 0
        } for tf in TF_CHOICES
    }

    pnl_log = []

    while True:
        for tf in TF_CHOICES:
            state = tf_state[tf]
            if state["cooldown"]:
                continue

            training = load_training(tf)
            if not training:
                continue

            k = MARKET.get_kline(f"{sym}-USDT", tf)[0]
            o, c = float(k[1]), float(k[2])
            candle = 100 * ((c - o) / o)

            res = evaluate(candle, training)
            if not res:
                state["decay"] += 1
                if state["decay"] >= DECAY_LIMIT:
                    state["cooldown"] = True
                continue

            move, conf = res
            size = size_multiplier(conf)

            if size == 0:
                continue

            # Backtest attribution (1 candle forward)
            if BACKTEST_MODE:
                next_k = MARKET.get_kline(f"{sym}-USDT", tf)[1]
                exit_price = float(next_k[2])
                entry = c
                pnl = (exit_price - entry) * math.copysign(1, move)

                pnl_log.append({
                    "tf": tf,
                    "conf": conf,
                    "size": size,
                    "pnl": pnl
                })

                if pnl > 0:
                    state["hits"] += 1
                    if state["hits"] >= RECOVERY_HITS:
                        state["decay"] = 0
                        state["cooldown"] = False

        if BACKTEST_MODE:
            with open("pnl_attribution.json", "w") as f:
                json.dump(pnl_log, f, indent=2)
            break

        time.sleep(0.2)

# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    coins = ["BTC", "ETH", "XRP", "BNB"]
    procs = []

    for c in coins:
        os.makedirs(coin_dir(c), exist_ok=True)
        p = mp.Process(target=thinker, args=(c,), daemon=True)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
