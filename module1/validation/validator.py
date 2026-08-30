"""
Dataset-Agnostic Canonical Data Validator.

Checks data integrity, timestamps, physiological bounds, duplicates,
and modality coverage, returning a structured ValidationReport.
"""

from collections import defaultdict
from datetime import timedelta
import logging
from typing import Dict, Any, List, Set, Tuple

from module1.schema.canonical_schema import (
    CanonicalPatientRecord,
    ModalityType,
    PHYSIOLOGICAL_BOUNDS,
)
from module1.validation.validation_report import (
    ValidationReport,
    ValidationIssue,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates a CanonicalPatientRecord against integrity and physiological rules.
    """

    def __init__(self, bounds_override: Dict[ModalityType, Tuple[float, float]] = None):
        self.bounds = bounds_override or PHYSIOLOGICAL_BOUNDS

    def validate(self, record: CanonicalPatientRecord) -> ValidationReport:
        """
        Execute full validation suite on the canonical patient record.
        """
        report = ValidationReport(
            patient_id=record.metadata.patient_id,
            dataset_source=record.metadata.dataset_source,
            total_events_checked=len(record.events),
        )

        # 1. Metadata Validation
        if not record.metadata.patient_id or record.metadata.patient_id == "unknown":
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="Metadata",
                message="Patient ID is missing or invalid.",
            ))

        if len(record.events) == 0:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="Completeness",
                message="Record contains zero events.",
            ))
            return report

        # 2. Chronological Ordering & Timestamp Validity
        prev_ts = None
        unordered_count = 0
        duplicate_event_keys: Dict[Tuple, int] = defaultdict(int)

        # Out-of-bounds counters by modality
        oob_counts: Dict[ModalityType, int] = defaultdict(int)
        oob_samples: Dict[ModalityType, float] = {}

        # Modality event counts
        modality_counts: Dict[ModalityType, int] = defaultdict(int)

        # CGM gap tracking
        prev_cgm_ts = None
        cgm_gaps_gt_2h = 0

        for e in record.events:
            modality_counts[e.modality] += 1

            # Chronological check
            if prev_ts and e.timestamp < prev_ts:
                unordered_count += 1
            prev_ts = e.timestamp

            # Duplicate check (exact same timestamp + modality + value)
            key = (e.timestamp, e.modality, e.value)
            duplicate_event_keys[key] += 1

            # Physiological bounds check
            if e.modality in self.bounds:
                min_b, max_b = self.bounds[e.modality]
                if e.value < min_b or e.value > max_b:
                    oob_counts[e.modality] += 1
                    if e.modality not in oob_samples:
                        oob_samples[e.modality] = e.value

            # CGM Gap tracking
            if e.modality == ModalityType.CGM_GLUCOSE:
                if prev_cgm_ts:
                    gap = (e.timestamp - prev_cgm_ts).total_seconds() / 3600.0
                    if gap > 2.0:
                        cgm_gaps_gt_2h += 1
                prev_cgm_ts = e.timestamp

        # Add unordered issues
        if unordered_count > 0:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Chronology",
                message=f"Detected {unordered_count} out-of-order events (will be sorted automatically).",
                count=unordered_count,
            ))

        # Add duplicate issues
        dup_count = sum(cnt - 1 for cnt in duplicate_event_keys.values() if cnt > 1)
        if dup_count > 0:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Duplicates",
                message=f"Found {dup_count} exact duplicate events.",
                count=dup_count,
            ))

        # Add Out-of-Bounds issues
        for mod, count in oob_counts.items():
            min_b, max_b = self.bounds[mod]
            sev = ValidationSeverity.ERROR if mod in (ModalityType.CGM_GLUCOSE, ModalityType.BOLUS_DOSE) else ValidationSeverity.WARNING
            report.add_issue(ValidationIssue(
                severity=sev,
                category="Physiological Bounds",
                message=f"{mod.value} had {count} values outside expected bounds [{min_b}, {max_b}]. Sample: {oob_samples[mod]}",
                count=count,
                sample_details={"sample_value": oob_samples[mod], "bounds": [min_b, max_b]},
            ))

        # Modality coverage checks
        if modality_counts[ModalityType.CGM_GLUCOSE] == 0:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="Modality Coverage",
                message="Critical modality 'cgm_glucose' is completely missing.",
            ))

        if modality_counts[ModalityType.BOLUS_DOSE] == 0 and modality_counts[ModalityType.BASAL_RATE] == 0:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Modality Coverage",
                message="No insulin administration records (bolus or basal) detected.",
            ))

        if cgm_gaps_gt_2h > 0:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Time Series Continuity",
                message=f"Detected {cgm_gaps_gt_2h} CGM measurement gaps exceeding 2 hours.",
                count=cgm_gaps_gt_2h,
            ))

        return report
