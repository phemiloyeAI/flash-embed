import contextlib
import time


@contextlib.contextmanager
def timer(name: str):
    start = time.time()
    yield
    duration = time.time() - start
    print(f"[timer] {name}: {duration:.4f}s")
