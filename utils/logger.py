# utils/logger.py
"""
Structured experiment logger — saves all metrics, configs, and events to JSON.
"""

import os
import json
import time
from datetime import datetime


class ExperimentLogger:
    """
    Logs experiment metadata, per-round metrics, and final results to a
    structured JSON file.
    """

    def __init__(self, log_dir: str, experiment_name: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{experiment_name}_{timestamp}.json")

        self.data = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "config": {},
            "events": [],
            "rounds": [],
            "final_metrics": {},
            "duration_seconds": 0,
        }
        self._start = time.time()

    def log_config(self, config: dict):
        """Store the full experiment configuration."""
        self.data["config"] = config

    def log_event(self, event: str, details: dict = None):
        """Log a timestamped event (e.g., 'training_started')."""
        entry = {
            "time": datetime.now().isoformat(),
            "event": event,
        }
        if details:
            entry["details"] = details
        self.data["events"].append(entry)
        print(f"  📝 {event}" + (f" | {details}" if details else ""))

    def log_round(self, round_num: int, metrics: dict, extra: dict = None):
        """Log metrics for a federated round or training epoch."""
        entry = {"round": round_num, "metrics": metrics}
        if extra:
            entry.update(extra)
        self.data["rounds"].append(entry)

    def log_final(self, metrics: dict):
        """Log final evaluation metrics."""
        self.data["final_metrics"] = metrics

    def save(self):
        """Write the log to disk."""
        self.data["duration_seconds"] = round(time.time() - self._start, 2)
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        print(f"  💾 Experiment log saved → {self.log_path}")
        return self.log_path
