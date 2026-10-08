"""
Runtime degradation tracking.

Collects every non-fatal degradation (fallback, missing data, skipped step)
during a pipeline run so it can be surfaced in results/pipeline_results.json
under a top-level "degraded_components" section, instead of hiding in logs.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DegradationTracker:
    """Accumulate degradation events during a pipeline run.

    Each event is a dict: {"component": str, "reason": str, "detail": ...}.
    Every record() also emits a WARNING log line, so degradations are visible
    both in real time (logs) and after the fact (results JSON).
    """

    def __init__(self):
        self.events: List[Dict] = []

    def record(self, component: str, reason: str, detail: str = "") -> None:
        event = {
            "component": component,
            "reason": reason,
            "detail": detail,
        }
        self.events.append(event)
        logger.warning(
            "DEGRADED [%s]: %s%s",
            component,
            reason,
            f" ({detail})" if detail else "",
        )

    def has(self, component: str) -> bool:
        return any(e["component"] == component for e in self.events)

    def to_dict(self) -> Dict:
        return {"degraded_components": list(self.events)}

    def __len__(self) -> int:
        return len(self.events)
