#!/usr/bin/env python3
"""
Cryptocurrency Pattern Recognition Trainer
Analyzes historical price patterns to predict future movements
"""

import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque

from kucoin.client import Market

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VERBOSE = False


@dataclass
class TradingConfig:
    """Configuration for trading parameters"""
    timeframes: List[str] = field(default_factory=lambda: ['1hour', '2hour', '4hour', '8hour', '12hour', '1day', '1week'])
    tf_minutes: List[int] = field(default_factory=lambda: [60, 120, 240, 480, 720, 1440, 10080])
    candle_count: int = 2
    candles_to_predict: int = 1
    max_history_lookback: int = 100000
    perfect_threshold_init: float = 1.0
    threshold_adjust_up: float = 0.01
    threshold_adjust_down: float = 0.01
    weight_adjustment: float = 0.25
    max_weight: float = 2.0
    min_weight: float = -2.0


@dataclass
class TradingMetrics:
    """Tracks trading performance metrics"""
    starting_amount: float = 100.0
    profit_list: List[float] = field(default_factory=list)
    good_predictions: List[float] = field(default_factory=list)
    accuracy_window: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_prediction(self, actual: float, predicted: float) -> None:
        """Track prediction accuracy"""
        error = abs((actual - predicted) / predicted) if predicted != 0 else 0
        self.accuracy_window.append(1 if error < 0.1 else 0)
    
    def get_accuracy(self) -> float:
        """Calculate current accuracy percentage"""
        return (sum(self.accuracy_window) / len(self.accuracy_window) * 100) if self.accuracy_window else 0.0


class MemoryManager:
    """Efficient in-memory cache for pattern memories and weights"""
    
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.memories: List[str] = []
        self.weights: List[float] = []
        self.high_weights: List[float] = []
        self.low_weights: List[float] = []
        self.dirty = False
        self.last_threshold = None
        self._load()
    
    def _get_path(self, suffix: str) -> Path:
        """Get file path for memory storage"""
        return Path(f"{suffix}_{self.timeframe}.txt")
    
    def _load(self) -> None:
        """Load memories from disk"""
        try:
            content = self._get_path("memories").read_text(encoding='utf-8', errors='ignore')
            self.memories = [m for m in content.replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split('~') if m.strip()]
        except FileNotFoundError:
            self.memories = []
        
        for attr, prefix in [('weights', 'memory_weights'), ('high_weights', 'memory_weights_high'), ('low_weights', 'memory_weights_low')]:
            try:
                content = self._get_path(prefix).read_text(encoding='utf-8', errors='ignore')
                values = [float(v) for v in content.replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split(' ') if v.strip()]
                setattr(self, attr, values)
            except (FileNotFoundError, ValueError):
                setattr(self, attr, [])
    
    def flush(self, force: bool = False) -> None:
        """Write changes to disk"""
        if not self.dirty and not force:
            return
        
        try:
            self._get_path("memories").write_text("~".join(self.memories), encoding='utf-8')
            self._get_path("memory_weights").write_text(" ".join(map(str, self.weights)), encoding='utf-8')
            self._get_path("memory_weights_high").write_text(" ".join(map(str, self.high_weights)), encoding='utf-8')
            self._get_path("memory_weights_low").write_text(" ".join(map(str, self.low_weights)), encoding='utf-8')
            self.dirty = False
        except Exception as e:
            logger.error(f"Error flushing memory: {e}")
    
    def add_pattern(self, pattern: List[float], high_diff: float, low_diff: float) -> None:
        """Add a new pattern to memory"""
        pattern_str = ' '.join(map(str, pattern))
        memory_entry = f"{pattern_str}{{}}{high_diff}{{}}{low_diff}"
        self.memories.append(memory_entry)
        self.weights.append(1.0)
        self.high_weights.append(1.0)
        self.low_weights.append(1.0)
        self.dirty = True
    
    def update_weights(self, indices: List[int], new_weights: List[float], 
                      new_high_weights: List[float], new_low_weights: List[float]) -> None:
        """Batch update weights"""
        for idx, w, hw, lw in zip(indices, new_weights, new_high_weights, new_low_weights):
            if 0 <= idx < len(self.weights):
                self.weights[idx] = w
                self.high_weights[idx] = hw
                self.low_weights[idx] = lw
        self.dirty = True
    
    def save_threshold(self, threshold: float, loop_iter: int, every: int = 200) -> None:
        """Periodically save threshold to reduce I/O"""
        if loop_iter % every != 0 and self.last_threshold is not None:
            if abs(threshold - self.last_threshold) < 0.05:
                return
        
        try:
            Path(f"neural_perfect_threshold_{self.timeframe}.txt").write_text(str(threshold), encoding='utf-8')
            self.last_threshold = threshold
        except Exception as e:
            logger.error(f"Error saving threshold: {e}")


class PatternMatcher:
    """Matches current patterns against historical data"""
    
    @staticmethod
    def calculate_difference(current: List[float], memory: List[float]) -> float:
        """Calculate percentage difference between two patterns"""
        if len(current) != len(memory):
            return float('inf')
        
        diffs = []
        for c, m in zip(current, memory):
            if c + m == 0:
                diffs.append(0.0)
            else:
                diffs.append(abs((abs(c - m) / ((c + m) / 2)) * 100))
        
        return sum(diffs) / len(diffs) if diffs else float('inf')
    
    @staticmethod
    def find_matches(current_pattern: List[float], memory_manager: MemoryManager, 
                    threshold: float) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
        """Find all patterns matching within threshold"""
        matches = {'indices': [], 'diffs': [], 'moves': [], 'high_moves': [], 'low_moves': []}
        
        for idx, mem_str in enumerate(memory_manager.memories):
            parts = mem_str.split('{}')
            if len(parts) < 3:
                continue
            
            pattern_str = parts[0].replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '')
            memory_pattern = [float(x) for x in pattern_str.split() if x.strip()]
            
            if len(memory_pattern) < len(current_pattern) + 1:
                continue
            
            diff = PatternMatcher.calculate_difference(current_pattern, memory_pattern[:-1])
            
            if diff <= threshold:
                matches['indices'].append(idx)
                matches['diffs'].append(diff)
                
                # Extract move predictions
                move = float(memory_pattern[-1])
                high_diff = float(parts[1].strip())
                low_diff = float(parts[2].strip())
                
                weight = memory_manager.weights[idx] if idx < len(memory_manager.weights) else 1.0
                high_weight = memory_manager.high_weights[idx] if idx < len(memory_manager.high_weights) else 1.0
                low_weight = memory_manager.low_weights[idx] if idx < len(memory_manager.low_weights) else 1.0
                
                matches['moves'].append(move * weight)
                matches['high_moves'].append(high_diff * high_weight)
                matches['low_moves'].append(low_diff * low_weight)
        
        return (matches['indices'], matches['diffs'], matches['moves'], 
                matches['high_moves'], matches['low_moves'])


class PriceDataProcessor:
    """Processes and manages price data"""
    
    @staticmethod
    def fetch_history(market: Market, coin: str, timeframe: str, 
                     start_time: int, end_time: int) -> List[str]:
        """Fetch historical candle data"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                history = str(market.get_kline(coin, timeframe, startAt=end_time, endAt=start_time))
                return history.replace(']]', '], ').replace('[[', '[').split('], [')
            except Exception as e:
                logger.warning(f"Fetch attempt {attempt + 1} failed: {e}")
                time.sleep(3.5)
        return []
    
    @staticmethod
    def parse_candles(history: List[str]) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Parse candle data into price lists"""
        opens, closes, highs, lows = [], [], [], []
        
        for candle_str in history:
            try:
                parts = candle_str.replace('"', '').replace("'", "").replace('[', '').split(", ")
                if len(parts) < 5:
                    continue
                
                opens.append(float(parts[1]))
                closes.append(float(parts[2]))
                highs.append(float(parts[3]))
                lows.append(float(parts[4]))
            except (ValueError, IndexError):
                continue
        
        # Reverse to get chronological order (oldest to newest)
        return [lst[::-1] for lst in [opens, closes, highs, lows]]
    
    @staticmethod
    def calculate_changes(prices: List[float], opens: List[float]) -> List[float]:
        """Calculate percentage changes from open to close"""
        changes = []
        for price, open_price in zip(prices, opens):
            if open_price != 0:
                change = 100 * ((price - open_price) / open_price)
                changes.append(change)
        return changes


class CryptoPatternTrainer:
    """Main training orchestrator"""
    
    def __init__(self, coin: str = "BTC"):
        self.coin = coin
        self.coin_pair = f"{coin}-USDT"
        self.market = Market(url='https://api.kucoin.com')
        self.config = TradingConfig()
        self.metrics = TradingMetrics()
        self.memory_managers: Dict[str, MemoryManager] = {}
        self.start_timestamp = int(time.time())
        
        self._write_status("TRAINING")
    
    def _write_status(self, state: str, finished_at: Optional[int] = None) -> None:
        """Write training status for GUI"""
        status = {
            "coin": self.coin,
            "state": state,
            "started_at": self.start_timestamp,
            "timestamp": int(time.time())
        }
        if finished_at:
            status["finished_at"] = finished_at
        
        try:
            Path("trainer_status.json").write_text(json.dumps(status, indent=2), encoding='utf-8')
        except Exception as e:
            logger.error(f"Error writing status: {e}")
    
    def _should_stop(self, loop_iter: int, check_every: int = 50) -> bool:
        """Check if training should stop"""
        if loop_iter % check_every != 0:
            return False
        
        try:
            killer_content = Path("killer.txt").read_text(encoding='utf-8', errors='ignore').strip().lower()
            return killer_content == "yes"
        except FileNotFoundError:
            return False
    
    def train_timeframe(self, timeframe: str, tf_minutes: int) -> None:
        """Train on a specific timeframe"""
        logger.info(f"Training on timeframe: {timeframe}")
        
        # Initialize memory manager
        memory_mgr = MemoryManager(timeframe)
        self.memory_managers[timeframe] = memory_mgr
        
        # Fetch historical data
        history = self._fetch_all_history(timeframe, tf_minutes)
        if not history:
            logger.error(f"No history data for {timeframe}")
            return
        
        # Parse price data
        opens, closes, highs, lows = PriceDataProcessor.parse_candles(history)
        if not closes:
            logger.error(f"Failed to parse candles for {timeframe}")
            return
        
        logger.info(f"Loaded {len(closes)} candles for {timeframe}")
        
        # Calculate percentage changes
        price_changes = PriceDataProcessor.calculate_changes(closes, opens)
        high_changes = PriceDataProcessor.calculate_changes(highs, opens)
        low_changes = PriceDataProcessor.calculate_changes(lows, opens)
        
        # Training loop
        perfect_threshold = self.config.perfect_threshold_init
        loop_iter = 0
        window_size = len(closes) // 2
        
        for current_idx in range(window_size, len(closes) - self.config.candles_to_predict):
            loop_iter += 1
            
            # Check for stop signal
            if self._should_stop(loop_iter):
                logger.info("Stop signal received")
                self._finalize_training(memory_mgr)
                return
            
            # Extract current pattern
            pattern_start = current_idx - self.config.candle_count + 1
            current_pattern = price_changes[pattern_start:current_idx]
            
            if len(current_pattern) < self.config.candle_count - 1:
                continue
            
            # Find matching patterns
            indices, diffs, moves, high_moves, low_moves = PatternMatcher.find_matches(
                current_pattern, memory_mgr, perfect_threshold
            )
            
            # Adjust threshold based on matches found
            if len(indices) > 20:
                perfect_threshold = max(0.0, perfect_threshold - self.config.threshold_adjust_down)
            else:
                perfect_threshold = min(100.0, perfect_threshold + self.config.threshold_adjust_up)
            
            # Save threshold periodically
            memory_mgr.save_threshold(perfect_threshold, loop_iter)
            
            # Calculate predictions
            if moves:
                avg_move = sum(moves) / len(moves)
                avg_high = sum(high_moves) / len(high_moves)
                avg_low = sum(low_moves) / len(low_moves)
            else:
                # No matches - add current pattern to memory
                future_idx = current_idx + self.config.candles_to_predict
                if future_idx < len(price_changes):
                    future_move = price_changes[future_idx]
                    future_high = high_changes[future_idx]
                    future_low = low_changes[future_idx]
                    
                    new_pattern = current_pattern + [future_move]
                    memory_mgr.add_pattern(new_pattern, future_high, future_low)
                continue
            
            # Update weights based on actual outcomes
            future_idx = current_idx + self.config.candles_to_predict
            if future_idx < len(price_changes):
                actual_move = price_changes[future_idx]
                actual_high = high_changes[future_idx]
                actual_low = low_changes[future_idx]
                
                self._update_weights(memory_mgr, indices, moves, high_moves, low_moves,
                                   actual_move, actual_high, actual_low)
            
            # Periodic flush
            if loop_iter % 200 == 0:
                memory_mgr.flush()
                logger.info(f"Progress: {current_idx}/{len(closes)} | Threshold: {perfect_threshold:.3f} | Matches: {len(indices)}")
        
        self._finalize_training(memory_mgr)
    
    def _update_weights(self, memory_mgr: MemoryManager, indices: List[int],
                       moves: List[float], high_moves: List[float], low_moves: List[float],
                       actual: float, actual_high: float, actual_low: float) -> None:
        """Update pattern weights based on prediction accuracy"""
        new_weights, new_high_weights, new_low_weights = [], [], []
        
        for idx, move, high_move, low_move in zip(indices, moves, high_moves, low_moves):
            current_weight = memory_mgr.weights[idx]
            current_high = memory_mgr.high_weights[idx]
            current_low = memory_mgr.low_weights[idx]
            
            # Adjust weight based on accuracy
            move_pct = (move / current_weight) * 100 if current_weight != 0 else 0
            
            # Standard weight adjustment
            if abs(actual - move_pct) < move_pct * 0.1:
                new_weight = min(self.config.max_weight, current_weight + self.config.weight_adjustment)
            else:
                new_weight = max(self.config.min_weight, current_weight - self.config.weight_adjustment)
            
            # Similar for high/low
            new_high = self._adjust_single_weight(current_high, high_move, actual_high)
            new_low = self._adjust_single_weight(current_low, low_move, actual_low)
            
            new_weights.append(new_weight)
            new_high_weights.append(new_high)
            new_low_weights.append(new_low)
        
        memory_mgr.update_weights(indices, new_weights, new_high_weights, new_low_weights)
    
    def _adjust_single_weight(self, current: float, predicted: float, actual: float) -> float:
        """Adjust a single weight value"""
        if abs(actual - predicted) < abs(predicted) * 0.1:
            return min(self.config.max_weight, current + self.config.weight_adjustment)
        else:
            return max(0.0, current - self.config.weight_adjustment)
    
    def _fetch_all_history(self, timeframe: str, tf_minutes: int) -> List[str]:
        """Fetch all available historical data"""
        all_history = []
        current_time = int(time.time())
        end_time = current_time - (1500 * tf_minutes * 60)
        
        # Try to load last start time
        try:
            last_start = int(Path("trainer_last_start_time.txt").read_text())
        except:
            last_start = 0
        
        logger.info(f"Fetching history for {timeframe}...")
        
        while current_time > last_start:
            batch = PriceDataProcessor.fetch_history(
                self.market, self.coin_pair, timeframe, current_time, end_time
            )
            
            if not batch or len(batch) < 100:
                break
            
            all_history.extend(batch)
            current_time = end_time
            end_time = current_time - (1500 * tf_minutes * 60)
            
            logger.info(f"Fetched {len(all_history)} total candles...")
            time.sleep(0.5)
        
        return all_history
    
    def _finalize_training(self, memory_mgr: MemoryManager) -> None:
        """Finalize training and save state"""
        memory_mgr.flush(force=True)
        
        finished_at = int(time.time())
        Path("trainer_last_training_time.txt").write_text(str(finished_at), encoding='utf-8')
        Path("trainer_last_start_time.txt").write_text(str(self.start_timestamp), encoding='utf-8')
        
        self._write_status("FINISHED", finished_at)
        logger.info(f"Training completed for {memory_mgr.timeframe}")
    
    def run(self) -> None:
        """Run full training across all timeframes"""
        logger.info(f"Starting training for {self.coin}")
        
        for timeframe, tf_minutes in zip(self.config.timeframes, self.config.tf_minutes):
            self.train_timeframe(timeframe, tf_minutes)
        
        logger.info("All timeframes processed successfully")


def main():
    """Entry point"""
    coin = "BTC"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        coin = sys.argv[1].strip().upper()
    
    logger.info(f"Initializing trainer for {coin}")
    
    trainer = CryptoPatternTrainer(coin)
    trainer.run()


if __name__ == "__main__":
    main()
