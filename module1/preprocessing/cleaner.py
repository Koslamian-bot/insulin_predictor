"""
Dataset-Agnostic Data Cleaner with Full Audit Trail.

Handles duplicate resolution, chronological enforcement, and corrupt/impossible
physiological value removal while maintaining an exact audit record.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from collections import defaultdict

from module1.schema.canonical_schema import (
    CanonicalPatientRecord,
    CanonicalEvent,
    ModalityType,
    PHYSIOLOGICAL_BOUNDS,
)

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    patient_id: str
    action: str
    modality: str
    timestamp: str
    value: Optional[float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CleaningAuditTrail:
    patient_id: str
    initial_event_count: int
    removed_event_count: int = 0
    final_event_count: int = 0
    reasons_summary: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    entries: List[AuditEntry] = field(default_factory=list)

    def log_removal(self, event: CanonicalEvent, action: str, reason: str) -> None:
        self.removed_event_count += 1
        self.reasons_summary[reason] += 1
        entry = AuditEntry(
            patient_id=self.patient_id,
            action=action,
            modality=event.modality.value,
            timestamp=event.timestamp.isoformat(),
            value=event.value,
            reason=reason,
        )
        self.entries.append(entry)

    def finalize(self, final_count: int) -> None:
        self.final_event_count = final_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "initial_event_count": self.initial_event_count,
            "removed_event_count": self.removed_event_count,
            "final_event_count": self.final_event_count,
            "reasons_summary": dict(self.reasons_summary),
            "total_audit_entries": len(self.entries),
        }

    def to_markdown(self) -> str:
        lines = [
            f"### Cleaning Audit Trail: Patient {self.patient_id}",
            f"- **Initial Events**: {self.initial_event_count:,}",
            f"- **Removed Events**: {self.removed_event_count:,}",
            f"- **Final Retained Events**: {self.final_event_count:,}",
            "",
        ]
        if self.reasons_summary:
            lines.append("| Removal Reason | Count |")
            lines.append("|---|---|")
            for reason, count in sorted(self.reasons_summary.items(), key=lambda x: -x[1]):
                lines.append(f"| {reason} | {count} |")
        else:
            lines.append("✓ Zero records required removal during cleaning.")
        return "\n".join(lines)


class DataCleaner:
    """
    Cleans canonical patient records and produces an immutable audit trail.
    """

    def __init__(self, bounds_override: Dict[ModalityType, Tuple[float, float]] = None):
        self.bounds = bounds_override or PHYSIOLOGICAL_BOUNDS

    def clean(self, record: CanonicalPatientRecord) -> Tuple[CanonicalPatientRecord, CleaningAuditTrail]:
        """
        Execute cleaning pipeline on the CanonicalPatientRecord.
        
        Returns:
            Tuple of (cleaned CanonicalPatientRecord, CleaningAuditTrail).
        """
        audit = CleaningAuditTrail(
            patient_id=record.metadata.patient_id,
            initial_event_count=len(record.events),
        )

        cleaned_events: List[CanonicalEvent] = []
        seen_exact_events: Set[Tuple[datetime, ModalityType, float]] = set()

        for event in record.events:
            # 1. Check exact duplicate
            key = (event.timestamp, event.modality, event.value)
            if key in seen_exact_events:
                audit.log_removal(
                    event=event,
                    action="DEDUPLICATE",
                    reason="Exact duplicate timestamp, modality, and value",
                )
                continue

            # 2. Check physiological bounds
            if event.modality in self.bounds:
                min_b, max_b = self.bounds[event.modality]
                if event.value < min_b or event.value > max_b:
                    audit.log_removal(
                        event=event,
                        action="FILTER_OUT_OF_BOUNDS",
                        reason=f"Value {event.value} outside physiological range [{min_b}, {max_b}]",
                    )
                    continue

            # Event is valid
            seen_exact_events.add(key)
            cleaned_events.append(event)

        # 3. Sort cleaned events chronologically
        cleaned_events.sort(key=lambda e: e.timestamp)

        audit.finalize(len(cleaned_events))

        cleaned_record = CanonicalPatientRecord(
            metadata=record.metadata,
            events=cleaned_events,
        )

        logger.info(
            "Cleaned patient %s: %d -> %d events (%d removed)",
            record.metadata.patient_id,
            audit.initial_event_count,
            audit.final_event_count,
            audit.removed_event_count,
        )

        return cleaned_record, audit
