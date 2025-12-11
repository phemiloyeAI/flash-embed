import enum
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


class TaskState(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RETRY = "retry"
    FAILED = "failed"
    DONE = "done"


@dataclass
class Task:
    uid: str
    state: TaskState = TaskState.PENDING
    retries: int = 0
    last_error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None


class Scheduler:
    """Task tracker for pipeline visibility and retries."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def start(self, uid: str) -> Task:
        task = self.tasks.get(uid, Task(uid=uid))
        task.state = TaskState.IN_PROGRESS
        task.started_at = time.time()
        self.tasks[uid] = task
        return task

    def complete(self, uid: str) -> Task:
        task = self.tasks[uid]
        task.state = TaskState.DONE
        task.ended_at = time.time()
        return task

    def fail(self, uid: str, error: str, retry: bool = False) -> Task:
        task = self.tasks.get(uid, Task(uid=uid))
        task.state = TaskState.RETRY if retry else TaskState.FAILED
        task.retries += 1 if retry else task.retries
        task.last_error = error
        task.ended_at = time.time()
        self.tasks[uid] = task
        return task
