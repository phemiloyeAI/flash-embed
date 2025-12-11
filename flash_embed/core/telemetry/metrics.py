class Metrics:
    """Placeholder metrics collector; replace with Prom/Grafana integration."""

    def __init__(self):
        self.counters = {}

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def observe(self, name: str, value: float) -> None:
        # histogram placeholder
        self.counters[name] = self.counters.get(name, 0) + value
