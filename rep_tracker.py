from dataclasses import dataclass
from typing import Literal

Stage = Literal["UP", "DOWN"]

@dataclass(frozen=True)
class RepState:
    count: int = 0
    stage: Stage = "UP"
    baseline_hip_angle: float | None = None

    form_errors: int = 0
    lowest_hip_angle: float = 180.0
    
    UP_THRESHOLD: float = 160.0
    DOWN_THRESHOLD: float = 90.0

def calculate_next_state(current_state: RepState, angle: float) -> RepState:
    count = current_state.count 
    stage = current_state.stage

    if stage == "UP":
        if angle < current_state.DOWN_THRESHOLD:
            print("DOWN Detected!")
            return RepState(count=count, stage="DOWN")
    elif stage == "DOWN":
        if angle > current_state.UP_THRESHOLD:
            print(f"UP Detected! Rep {count + 1}!")
            return RepState(count=count+1, stage="UP")
    return current_state


