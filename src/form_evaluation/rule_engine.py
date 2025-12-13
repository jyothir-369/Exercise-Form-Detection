# src/form_evaluation/rule_engine.py
from .bicep_curl_rules import BicepCurlRules
from .lateral_raise_rules import LateralRaiseRules
from .posture_rules import PostureRules


class RuleEngine:
    def __init__(self):
        self.rules = {
            "bicep_curl": BicepCurlRules(),
            "lateral_raise": LateralRaiseRules(),
            "posture": PostureRules(),
        }

    def evaluate(self, exercise_type: str, keypoints_2d):
        if exercise_type not in self.rules:
            return {
                "status": "error",
                "message": f"Unknown exercise: {exercise_type}",
                "angle": None
            }

        # ← FIXED: normal dot, not Chinese period
        result = self.rules[exercise_type].evaluate(keypoints_2d)

        # Safety net: convert list → dict if someone returns wrong format
        if isinstance(result, list):
            if len(result) >= 2:
                return {
                    "status": result[0],
                    "message": result[1],
                    "angle": result[2] if len(result) > 2 else None
                }
            return {"status": "unknown", "message": "Invalid rule output"}

        if isinstance(result, dict):
            return result

        return {"status": "unknown", "message": str(result), "angle": None}