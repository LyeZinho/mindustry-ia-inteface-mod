import pytest
from mindustry_ai.game.action_executor import ActionExecutor


def test_executor_init():
    executor = ActionExecutor()
    assert executor is not None


def test_place_drill_action():
    executor = ActionExecutor()
    result = executor.execute(action=0, x=100, y=100)
    assert result is True


def test_place_conveyor_action():
    executor = ActionExecutor()
    result = executor.execute(action=1, x=100, y=100)
    assert result is True


def test_wait_action():
    executor = ActionExecutor()
    result = executor.execute(action=6)
    assert result is True
