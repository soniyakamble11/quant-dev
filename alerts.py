import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List
import queue
from datetime import datetime, timezone

@dataclass
class Alert:
    timestamp: int
    metric: str
    value: float
    rule_text: str

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "time_iso": datetime.fromtimestamp(self.timestamp/1000, tz=timezone.utc).isoformat(),
            "metric": self.metric,
            "value": self.value,
            "rule": self.rule_text
        }

class AlertEngine:
    def __init__(self, poll_interval=0.5):
        self.poll_interval = poll_interval
        self.alert_queue = queue.Queue()
        self.rules = []
        self.running = False
        self.thread = None

    def register_rule(self, rule_fn: Callable):
        """rule_fn must return: (fired: bool, metric_name, metric_value, rule_text)"""
        self.rules.append(rule_fn)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _loop(self):
        while self.running:
            for rule_fn in self.rules:
                try:
                    fired, metric, value, rule_text = rule_fn()
                    if fired:
                        alert = Alert(
                            timestamp=int(time.time() * 1000),
                            metric=metric,
                            value=float(value),
                            rule_text=rule_text
                        )
                        self.alert_queue.put(alert)
                        self._log(alert)
                except Exception as e:
                    print("Alert rule error:", e)
            time.sleep(self.poll_interval)

    def _log(self, alert: Alert):
        with open("logs/live_alerts.log", "a") as f:
            f.write(str(alert.to_dict()) + "\n")

    def get_recent_alerts(self, n=5) -> List[Dict[str, Any]]:
        """Non-destructive read of newest alerts"""
        items = []
        temp = []
        try:
            while True:
                al = self.alert_queue.get_nowait()
                items.append(al)
                temp.append(al)
        except queue.Empty:
            pass
        # requeue
        for x in temp:
            self.alert_queue.put(x)
        # return last n
        return [a.to_dict() for a in items[-n:]]
