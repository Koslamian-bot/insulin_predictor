"""
Validation Report Data Structure and Severity Levels.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json


class ValidationSeverity(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    category: str
    message: str
    count: int = 1
    sample_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "count": self.count,
            "sample_details": self.sample_details,
        }


@dataclass
class ValidationReport:
    patient_id: str
    dataset_source: str
    total_events_checked: int
    issues: List[ValidationIssue] = field(default_factory=list)
    overall_status: ValidationSeverity = ValidationSeverity.PASS

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.overall_status = ValidationSeverity.ERROR
        elif issue.severity == ValidationSeverity.WARNING and self.overall_status != ValidationSeverity.ERROR:
            self.overall_status = ValidationSeverity.WARNING

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "dataset_source": self.dataset_source,
            "total_events_checked": self.total_events_checked,
            "overall_status": self.overall_status.value,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
        }

    def to_markdown(self) -> str:
        lines = [
            f"### Validation Report: Patient {self.patient_id} ({self.dataset_source})",
            f"- **Overall Status**: `{self.overall_status.value}`",
            f"- **Total Events Evaluated**: {self.total_events_checked:,}",
            f"- **Errors**: {self.error_count}",
            f"- **Warnings**: {self.warning_count}",
            "",
        ]
        if self.issues:
            lines.append("| Severity | Category | Description | Count |")
            lines.append("|---|---|---|---|")
            for issue in self.issues:
                lines.append(f"| `{issue.severity.value}` | {issue.category} | {issue.message} | {issue.count} |")
        else:
            lines.append("✓ All validation checks passed cleanly with zero issues.")
        return "\n".join(lines)
