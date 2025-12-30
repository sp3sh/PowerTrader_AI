#!/usr/bin/env python3
"""
PowerTrader AI – Unified Trainer + Thinker + Backtester
Authoritative, cleaned, deterministic version
"""

import time
import sys
import math
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("PowerTrader")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

@dataclass
class Config:
    candle_count: int = 2
    threshold_init: float = 1.0
    threshold_up: float = 0.01
    threshold_down: float = 0.01

    weight_step: float = 0.25
    max_weight: float = 2.0
    min_weight: float = -2.0

    min_matches: int = 5
    prune_weight_below: float = 0.25
    prune_every: int = 5000

# --------------------------------------------------
# MEMORY PARSING
# --------------------------------------------------

def parse_memory(raw: str):
    """
    <p1 p2 ... future_move>{}high{}low
    """
    parts = raw.split("{}")
    pat = [float(x) for x in parts[0].split()]
    high = float(parts[1]) / 100
    low = float(parts[2]) / 100
    return pat[:-1], pat[-1], high, low

# --------------------------------------------------
# DIFFERENCE METRIC
# --------------------------------------------------

def diff(a: List[float], b: List[float]) -> float:
    d = 0.0
    for x, y in zip(a, b):
        if x + y == 0:
            continue
        d += abs((x - y) / ((x + y) / 2)) * 100
    return d / len(a)

# --------------------------------------------------
# MEMORY BANK (THINKER CORE)
# --------------------------------------------------

class MemoryBank:
    def __init__(self, tf: str):
        self.tf = tf
        self.patterns = []
        self.moves = []
        self.highs = []
        self.lows = []
        self.weights = []
        self.load()

    def load(self):
        mems = Path(f"memories_{self.tf}.txt").read_text().split("~")
        ws = list(map(float, Path(f"memory_weights_{self.tf}.txt").read_text().split()))
        hws = list(map(float, Path(f"memory_weights_high_{self.tf}.txt").read_text().split()))
        lws = list(map(float, Path(f"memory_weights_low_{self.tf}.txt").read_text().split()))

        for raw, w, hw, lw in zip(mems, ws, hws, lws):
            p, m, h, l = parse_memory(raw)
            self.patterns.append(p)
            self.moves.append(m * w)
            self.highs.append(h * hw)
            self.lows.append(l * lw)
            self.weights.append(w)

# --------------------------------------------------
# THINKER (PREDICTION)
# --------------------------------------------------

def predict(pattern: List[float], bank: MemoryBank, threshold: float):
    matches = []
    for p, m, h, l, w in zip(
        bank.patterns, bank.moves, bank.highs, bank.lows, bank.weights
    ):
        if diff(pattern, p) <= threshold:
            matches.append((m, h, l, abs(w)))

    if len(matches) < 1:
        return None

    move = sum(m for m, _, _, w in matches) / len(matches)
    high = sum(h for _, h, _, _ in matches) / len(matches)
    low = sum(l for _, _, l, _ in matches) / len(matches)

    confidence = min(
        1.0,
        math.log(len(matches) + 1)
        * (sum(w for *_, w in matches) / len(matches))
        / 5,
    )

    return {
        "move": move,
        "high": high,
        "low": low,
        "confidence": round(confidence, 3),
        "matches": len(matches),
    }

# --------------------------------------------------
# MEMORY PRUNING
# --------------------------------------------------

def prune_memory(tf: str, cfg: Config):
    mem = Path(f"memories_{tf}.txt").read_text().split("~")
    w = list(map(float, Path(f"memory_weights_{tf}.txt").read_text().split()))
    hw = list(map(float, Path(f"memory_weights_high_{tf}.txt").read_text().split()))
    lw = list(map(float, Path(f"memory_weights_low_{tf}.txt").read_text().split()))

    kept = [
        (m, x, y, z)
        for m, x, y, z in zip(mem, w, hw, lw)
        if abs(x) >= cfg.prune_weight_below
    ]

    if not kept:
        return

    mem, w, hw, lw = zip(*kept)

    Path(f"memories_{tf}.txt").write_text("~".join(mem))
    Path(f"memory_weights_{tf}.txt").write_text(" ".join(map(str, w)))
    Path(f"memory_weights_high_{tf}.txt").write_text(" ".join(map(str, hw)))
    Path(f"memory_weights_low_{tf}.txt").write_text(" ".join(map(str, lw)))

    log.info(f"[{tf}] pruned to {len(mem)} memories")

# --------------------------------------------------
# BACKTEST HARNESS
# --------------------------------------------------

def backtest(tf: str, price_changes: List[float], threshold: float):
    bank = MemoryBank(tf)
    wins = 0
    total = 0

    for i in range(1, len(price_changes) - 1):
        pat = price_changes[i - 1:i]
        out = predict(pat, bank, threshold)
        if not out:
            continue

        actual = price_changes[i + 1]
        total += 1
        if abs(out["move"] - actual) < abs(actual) * 0.1:
            wins += 1

    return wins / total if total else 0

# --------------------------------------------------
# MULTIPROCESS RUNNER
# --------------------------------------------------

def run_parallel(func, items):
    with mp.Pool(mp.cpu_count()) as pool:
        return pool.map(func, items)

# --------------------------------------------------
# ENTRY
# --------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python powertrader_core.py <timeframe>")
        sys.exit(1)

    tf = sys.argv[1]
    cfg = Config()

    log.info(f"Loading memory bank for {tf}")
    bank = MemoryBank(tf)

    # Example inference
    example_pattern = [0.5]  # % change
    out = predict(example_pattern, bank, cfg.threshold_init)

    print(json.dumps(out, indent=2))

    # Optional prune
    prune_memory(tf, cfg)
