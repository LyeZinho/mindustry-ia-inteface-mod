from typing import Dict, Any
from mindustry_ai.rules.behavior_tree import BehaviorTree, Action
from mindustry_ai.rules.state_machine import StateMachine
from mindustry_ai.rules.priority_queue import PriorityQueue


class HybridDecider:
    def __init__(self):
        self.behavior_tree = BehaviorTree()
        self.state_machine = StateMachine()
        self.priority_queue = PriorityQueue()
    
    def decide(self, state: Dict[str, Any]) -> int:
        # 1. Behavior Tree: get feasible actions
        feasible_actions = self.behavior_tree.get_feasible_actions(state)
        
        if not feasible_actions:
            return Action.WAIT
        
        # If only one action available, take it
        if len(feasible_actions) == 1:
            return feasible_actions[0]
        
        # 2. State Machine: get current operational state
        current_game_state = self.state_machine.update(state)
        
        # 3. Priority Queue: compute urgency
        priorities = self.priority_queue.compute_priorities(state)
        highest_priority = self.priority_queue.get_highest_priority_category(priorities)
        
        # 4. Map priority category to action
        priority_to_action = {
            "survive": Action.PLACE_TURRET,
            "power": Action.PLACE_GENERATOR,
            "mining": Action.PLACE_DRILL,
            "optimize": Action.PLACE_CONVEYOR,
        }
        
        preferred_action = priority_to_action[highest_priority]
        
        # 5. Return preferred action if feasible, else pick first feasible
        if preferred_action in feasible_actions:
            return preferred_action
        else:
            return feasible_actions[0]
