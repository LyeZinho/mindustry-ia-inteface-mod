from typing import Dict, Any


class PriorityQueue:
    def compute_priorities(self, state: Dict[str, Any]) -> Dict[str, float]:
        priorities = {}
        
        # Survival priority
        enemies_nearby = state["threat"]["enemies_nearby"]
        threat_level = min(1.0, enemies_nearby / 5.0)
        priorities["survive"] = threat_level * 100
        
        # Power priority
        power_ratio = state["power"]["current"] / state["power"]["capacity"]
        power_deficit = max(0, 1.0 - power_ratio)
        priorities["power"] = power_deficit * 50
        
        # Mining priority
        copper_target = 200
        copper_ratio = state["resources"]["copper"] / copper_target
        mining_deficit = max(0, 1.0 - copper_ratio)
        priorities["mining"] = mining_deficit * 30
        
        # Optimization priority (always present, lowest)
        priorities["optimize"] = 10
        
        return priorities
    
    def get_highest_priority_category(self, priorities: Dict[str, float]) -> str:
        return max(priorities, key=priorities.get)
