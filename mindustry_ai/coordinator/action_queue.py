import threading
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Action:
    type: str
    position: Tuple[int, int]
    source: str
    timestamp: float


class ActionQueue:
    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()
    
    def enqueue(self, action: Action) -> None:
        with self._lock:
            self._queue.append(action)
    
    def dequeue(self) -> Optional[Action]:
        with self._lock:
            if len(self._queue) == 0:
                return None
            return self._queue.pop(0)
    
    def peek(self) -> Optional[Action]:
        with self._lock:
            if len(self._queue) == 0:
                return None
            return self._queue[0]
    
    def size(self) -> int:
        with self._lock:
            return len(self._queue)
    
    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
