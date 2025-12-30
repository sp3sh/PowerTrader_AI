#!/usr/bin/env python3
"""
PowerTrader AI – Trainer
Versioned, confidence-aware pattern trainer with pruning and backtest attribution
"""

import sys
import json
import time
import math
import logging
import hashlib
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict

from kucoin.client import Market

# =========================
# GLOBAL VERSION CONTRACT
# =========================

TRAINER_VERSION = "2.2.0"
MEMORY_SCHEMA_VERSION = "1.1"

VERSION_CONTRACT = {
    "trainer_version": TRAINER_VERSION,
    "memory_schema": MEMORY_SCHEMA_VERSION,
    "required_thinker_min": "2.2.0"
}

# =========================
# LOGGING
# =========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trainer")

# =========================
# CONFIGURATION
# =========================

@dataclass
class TradingConfig:
    timeframes: List[str] = field(default_factory=lambda: [
        "1hour", "2hour", "4hour", "8hour", "12hour", "1day"
    ])
    tf_minutes: List[int] = field(default_factory=lambda: [
        60, 120, 240, 480, 720, 1440
    ])

    candle_count: int = 2
    candles_to_predict: int = 1

    perfect_threshold_init: float = 1.0
    threshold_adjust_up: float = 0.02
    threshold_adjust_down: float = 0.02

    weight_adjustment: float = 0.15
    max_weight: float = 2.0
    min_weight: float = -1.5

    memory_prune_below_weight: float = -0.5
    prune_every: int = 1000

    confidence_decay: float = 0.995
    disable_tf_confidence: float = 0.45

    max_memory: int = 150_000

# =========================
# MEMORY MANAGER
# =========================

class MemoryManager:
    def __init__(self, timeframe: str):
        self.tf = timeframe
        self.memories: List[str] = []
        self.weights: List[float] = []
        self.confidence: List[float] = []
        self.dirty = False
        self._load()

    def _path(self, name: str) -> Path:
        return Path(f"{name}_{self.tf}.txt")

    def _load(self):
        try:
            self.memories = self._path("memories").read_text().split("~")
            self.weights = list(map(float, self._path("weights").read_text().split()))
            self.confidence = list(map(float, self._path("confidence").read_text().split()))
        except Exception:
            self.memories, self.weights, self.confidence = [], [], []

    def flush(self, force=False):
        if not self.dirty and not force:
            return
        self._path("memories").write_text("~".join(self.memories))
        self._path("weights").write_text(" ".join(map(str, self.weights)))
        self._path("confidence").write_text(" ".join(map(str, self.confidence)))
        self.dirty = False

    def add(self, pattern: List[float], future_move: float):
        entry = " ".join(map(str, pattern + [future_move]))
        self.memories.append(entry)
        self.weights.append(1.0)
        self.confidence.append(1.0)
        self.dirty = True

    def prune(self, config: TradingConfig):
        keep = [
            i for i, w in enumerate(self.weights)
            if w >= config.memory_prune_below_weight
        ]
        self.memories = [self.memories[i] for i in keep]
        self.weights = [self.weights[i] for i in keep]
        self.confidence = [self.confidence[i] for i in keep]
        self.dirty = True

# =========================
# PATTERN MATCHER
# =========================

def diff(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

def match_patterns(pattern, memory_mgr, threshold):
    idxs, preds, confs = [], [], []
    for i, mem in enumerate(memory_mgr.memories):
        values = list(map(float, mem.split()))
        past, future = values[:-1], values[-1]
        if len(past) != len(pattern):
            continue
        if diff(pattern, past) <= threshold:
            idxs.append(i)
            preds.append(future * memory_mgr.weights[i])
            confs.append(memory_mgr.confidence[i])
    return idxs, preds, confs

# =========================
# TRAINER
# =========================

class CryptoPatternTrainer:
    def __init__(self, coin="BTC"):
        self.coin = coin
        self.pair = f"{coin}-USDT"
        self.market = Market()
        self.cfg = TradingConfig()
        self.tf_confidence = defaultdict(lambda: 1.0)
        self.disabled_tfs = set()

        self._write_version_contract()

    def _write_version_contract(self):
        Path("trainer_version.json").write_text(json.dumps(VERSION_CONTRACT, indent=2))

    def train_tf(self, tf, tf_min):
        if tf in self.disabled_tfs:
            return

        mem = MemoryManager(tf)
        candles = self._load_history(tf)
        changes = candles

        threshold = self.cfg.perfect_threshold_init

        for i in range(self.cfg.candle_count, len(changes) - 1):
            pattern = changes[i - self.cfg.candle_count:i]
            idxs, preds, confs = match_patterns(pattern, mem, threshold)

            if preds:
                prediction = sum(preds) / len(preds)
                confidence = sum(confs) / len(confs)
                actual = changes[i + 1]

                error = abs(actual - prediction)
                hit = error < abs(prediction) * 0.1

                for idx in idxs:
                    mem.weights[idx] += self.cfg.weight_adjustment if hit else -self.cfg.weight_adjustment
                    mem.confidence[idx] *= self.cfg.confidence_decay

                self.tf_confidence[tf] *= self.cfg.confidence_decay
                if not hit:
                    self.tf_confidence[tf] *= 0.98

            else:
                mem.add(pattern, changes[i + 1])

            if i % self.cfg.prune_every == 0:
                mem.prune(self.cfg)
                mem.flush()

            if self.tf_confidence[tf] < self.cfg.disable_tf_confidence:
                logger.warning(f"Disabling TF {tf} due to confidence decay")
                self.disabled_tfs.add(tf)
                break

        mem.flush(force=True)

    def _load_history(self, tf):
        # simplified mock – replace with kucoin loader
        return []

    def run(self):
        for tf, tf_min in zip(self.cfg.timeframes, self.cfg.tf_minutes):
            self.train_tf(tf, tf_min)

# =========================
# ENTRY
# =========================

def main():
    coin = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    trainer = CryptoPatternTrainer(coin)
    trainer.run()

if __name__ == "__main__":
    main()
