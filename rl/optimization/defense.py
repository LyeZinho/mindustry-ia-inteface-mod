import math

TURRET_RANGES = {
    "duo": 10,
    "scatter": 9,
    "hail": 13,
    "wave": 8,
    "lancer": 14,
    "arc": 10,
    "swarmer": 12,
}

def compute_defense_gap(state: dict) -> float:
    buildings = state.get("buildings", [])
    
    turrets = []
    for b in buildings:
        block_name = b.get("block", "")
        if block_name in TURRET_RANGES:
            turrets.append({
                "x": b.get("x", 0),
                "y": b.get("y", 0),
                "range": TURRET_RANGES[block_name],
            })
    
    if not turrets:
        return 1.0
    
    core = state.get("core", {})
    core_x = core.get("x", 0)
    core_y = core.get("y", 0)
    core_size = core.get("size", 3)
    
    perimeter_radius = core_size + 3
    num_samples = 24
    uncovered_count = 0
    
    for i in range(num_samples):
        angle = (i * 360.0 / num_samples) * (math.pi / 180.0)
        px = core_x + perimeter_radius * math.cos(angle)
        py = core_y + perimeter_radius * math.sin(angle)
        
        covered = False
        for turret in turrets:
            dist = math.sqrt((px - turret["x"])**2 + (py - turret["y"])**2)
            if dist <= turret["range"]:
                covered = True
                break
        
        if not covered:
            uncovered_count += 1
    
    return uncovered_count / num_samples
