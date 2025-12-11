from typing import Iterable

class Downloader:
    """Placeholder downloader; extend with remote fetch + retries."""

    def fetch(self, urls: Iterable[str]) -> None:
        raise NotImplementedError("Implement remote fetch logic")
