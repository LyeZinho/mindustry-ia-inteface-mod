def compute_power_deficit(state: dict) -> float:
    power = state.get("power", {})
    produced = float(power.get("produced", 0.0))
    consumed = float(power.get("consumed", 0.0))
    capacity = float(power.get("capacity", 1.0))
    
    deficit = max(0.0, consumed - produced)
    
    return min(1.0, deficit / max(capacity, 1.0))
