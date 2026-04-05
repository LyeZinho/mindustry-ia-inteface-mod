import pytest
from dataclasses import dataclass
from mindustry_ai.coordinator.action_queue import ActionQueue, Action


class TestActionQueue:
    def test_enqueue_dequeue_single_action(self):
        queue = ActionQueue()
        action = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        
        queue.enqueue(action)
        assert queue.size() == 1
        
        dequeued = queue.dequeue()
        assert dequeued is not None
        assert dequeued.source == "human"
        assert dequeued.type == "PLACE_DRILL"
        assert queue.size() == 0
    
    def test_fifo_order(self):
        queue = ActionQueue()
        a1 = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        a2 = Action(type="PLACE_CONVEYOR", position=(6, 5), source="ai", timestamp=0.5)
        
        queue.enqueue(a1)
        queue.enqueue(a2)
        
        first = queue.dequeue()
        second = queue.dequeue()
        assert first is not None
        assert second is not None
        assert first.source == "human"
        assert second.source == "ai"
    
    def test_peek_does_not_remove(self):
        queue = ActionQueue()
        action = Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0)
        
        queue.enqueue(action)
        peeked = queue.peek()
        
        assert peeked is not None
        assert peeked.type == "PLACE_DRILL"
        assert queue.size() == 1
    
    def test_empty_queue_dequeue_returns_none(self):
        queue = ActionQueue()
        assert queue.dequeue() is None
    
    def test_clear_empties_queue(self):
        queue = ActionQueue()
        queue.enqueue(Action(type="PLACE_DRILL", position=(5, 5), source="human", timestamp=0.0))
        queue.enqueue(Action(type="PLACE_CONVEYOR", position=(6, 5), source="ai", timestamp=0.5))
        
        queue.clear()
        assert queue.size() == 0
