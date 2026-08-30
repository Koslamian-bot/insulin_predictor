"""
Canonical Schema for Diabetes Multi-Modal Time-Series Data.

Defines dataset-agnostic data models, standard physiological units,
and bounded validation parameters.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
import json
import pandas as pd


class ModalityType(str, Enum):
    """Enumeration of all supported physiological and behavioral modalities."""
    CGM_GLUCOSE = "cgm_glucose"
    FINGERSTICK_GLUCOSE = "fingerstick_glucose"
    BASAL_RATE = "basal_rate"
    TEMP_BASAL_RATE = "temp_basal_rate"
    BOLUS_DOSE = "bolus_dose"
    MEAL_CARBS = "meal_carbs"
    HEART_RATE = "heart_rate"
    SKIN_TEMPERATURE = "skin_temperature"
    AIR_TEMPERATURE = "air_temperature"
    GSR = "gsr"
    STEPS = "steps"
    ACCELERATION = "acceleration"
    SLEEP = "sleep"
    EXERCISE = "exercise"
    WORK = "work"
    STRESS = "stress"
    ILLNESS = "illness"
    HYPO_EVENT = "hypo_event"


# Standard Project Units
CANONICAL_UNITS: Dict[ModalityType, str] = {
    ModalityType.CGM_GLUCOSE: "mg/dL",
    ModalityType.FINGERSTICK_GLUCOSE: "mg/dL",
    ModalityType.BASAL_RATE: "U/hr",
    ModalityType.TEMP_BASAL_RATE: "U/hr",
    ModalityType.BOLUS_DOSE: "Units",
    ModalityType.MEAL_CARBS: "grams",
    ModalityType.HEART_RATE: "BPM",
    ModalityType.SKIN_TEMPERATURE: "degF",
    ModalityType.AIR_TEMPERATURE: "degF",
    ModalityType.GSR: "Siemens",
    ModalityType.STEPS: "count",
    ModalityType.ACCELERATION: "g",
    ModalityType.SLEEP: "quality_score",
    ModalityType.EXERCISE: "intensity_score",
    ModalityType.WORK: "intensity_score",
    ModalityType.STRESS: "event_flag",
    ModalityType.ILLNESS: "event_flag",
    ModalityType.HYPO_EVENT: "event_flag",
}

# Physiological Validation Bounds (Min, Max)
PHYSIOLOGICAL_BOUNDS: Dict[ModalityType, Tuple[float, float]] = {
    ModalityType.CGM_GLUCOSE: (20.0, 600.0),          # mg/dL
    ModalityType.FINGERSTICK_GLUCOSE: (20.0, 600.0),    # mg/dL
    ModalityType.BASAL_RATE: (0.0, 15.0),              # U/hr
    ModalityType.TEMP_BASAL_RATE: (0.0, 15.0),         # U/hr
    ModalityType.BOLUS_DOSE: (0.0, 60.0),              # Units
    ModalityType.MEAL_CARBS: (0.0, 500.0),             # grams
    ModalityType.HEART_RATE: (30.0, 240.0),            # BPM
    ModalityType.SKIN_TEMPERATURE: (65.0, 115.0),      # degF
    ModalityType.AIR_TEMPERATURE: (-20.0, 140.0),      # degF
    ModalityType.GSR: (0.0, 1.0),                      # Siemens / uS range
    ModalityType.STEPS: (0.0, 5000.0),                 # steps per interval
    ModalityType.ACCELERATION: (0.0, 20.0),            # g
    ModalityType.SLEEP: (0.0, 100.0),                  # score / quality
    ModalityType.EXERCISE: (0.0, 10.0),                # intensity score
    ModalityType.WORK: (0.0, 10.0),                    # intensity score
}


@dataclass
class ModalityInfo:
    """Metadata describing a modality."""
    modality: ModalityType
    is_continuous: bool
    unit: str
    description: str


MODALITY_METADATA: Dict[ModalityType, ModalityInfo] = {
    ModalityType.CGM_GLUCOSE: ModalityInfo(ModalityType.CGM_GLUCOSE, True, "mg/dL", "Continuous Glucose Monitor reading"),
    ModalityType.FINGERSTICK_GLUCOSE: ModalityInfo(ModalityType.FINGERSTICK_GLUCOSE, False, "mg/dL", "Finger-stick capillary blood glucose"),
    ModalityType.BASAL_RATE: ModalityInfo(ModalityType.BASAL_RATE, True, "U/hr", "Scheduled continuous basal insulin rate"),
    ModalityType.TEMP_BASAL_RATE: ModalityInfo(ModalityType.TEMP_BASAL_RATE, True, "U/hr", "Temporary basal insulin rate override"),
    ModalityType.BOLUS_DOSE: ModalityInfo(ModalityType.BOLUS_DOSE, False, "Units", "Rapid-acting bolus insulin delivery"),
    ModalityType.MEAL_CARBS: ModalityInfo(ModalityType.MEAL_CARBS, False, "grams", "Carbohydrate intake from meals"),
    ModalityType.HEART_RATE: ModalityInfo(ModalityType.HEART_RATE, True, "BPM", "Heart rate from wristband sensor"),
    ModalityType.SKIN_TEMPERATURE: ModalityInfo(ModalityType.SKIN_TEMPERATURE, True, "degF", "Skin temperature from wearable"),
    ModalityType.AIR_TEMPERATURE: ModalityInfo(ModalityType.AIR_TEMPERATURE, True, "degF", "Ambient air temperature"),
    ModalityType.GSR: ModalityInfo(ModalityType.GSR, True, "Siemens", "Galvanic skin response / skin conductance"),
    ModalityType.STEPS: ModalityInfo(ModalityType.STEPS, True, "count", "Step count in interval"),
    ModalityType.ACCELERATION: ModalityInfo(ModalityType.ACCELERATION, True, "g", "3-axis accelerometer magnitude"),
    ModalityType.SLEEP: ModalityInfo(ModalityType.SLEEP, False, "score", "Sleep period and quality assessment"),
    ModalityType.EXERCISE: ModalityInfo(ModalityType.EXERCISE, False, "score", "Self-reported physical exercise"),
    ModalityType.WORK: ModalityInfo(ModalityType.WORK, False, "score", "Self-reported work period"),
    ModalityType.STRESS: ModalityInfo(ModalityType.STRESS, False, "flag", "Self-reported stress event"),
    ModalityType.ILLNESS: ModalityInfo(ModalityType.ILLNESS, False, "flag", "Self-reported illness period"),
    ModalityType.HYPO_EVENT: ModalityInfo(ModalityType.HYPO_EVENT, False, "flag", "Self-reported hypoglycemia symptom event"),
}


@dataclass
class CanonicalEvent:
    """
    Standardized, atomic event representation across all datasets.
    """
    timestamp: datetime
    modality: ModalityType
    value: float
    value_secondary: Optional[float] = None
    timestamp_end: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "modality": self.modality.value,
            "value": self.value,
            "value_secondary": self.value_secondary,
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalEvent":
        ts = datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        ts_end = None
        if data.get("timestamp_end"):
            ts_end = datetime.fromisoformat(data["timestamp_end"]) if isinstance(data["timestamp_end"], str) else data["timestamp_end"]
        return cls(
            timestamp=ts,
            modality=ModalityType(data["modality"]),
            value=float(data["value"]),
            value_secondary=float(data["value_secondary"]) if data.get("value_secondary") is not None else None,
            timestamp_end=ts_end,
            attributes=data.get("attributes", {}),
        )


@dataclass
class PatientMetadata:
    """Metadata describing the patient and provenance."""
    patient_id: str
    dataset_source: str
    cohort: Optional[str] = None
    split_label: Optional[str] = None
    weight_kg: Optional[float] = None
    insulin_type: Optional[str] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientMetadata":
        return cls(**data)


@dataclass
class CanonicalPatientRecord:
    """
    Complete canonical container for a single patient file / stream.
    Contains metadata and a chronologically sorted sequence of CanonicalEvents.
    """
    metadata: PatientMetadata
    events: List[CanonicalEvent] = field(default_factory=list)

    def add_event(self, event: CanonicalEvent) -> None:
        self.events.append(event)

    def sort_events(self) -> None:
        """Sort all events chronologically."""
        self.events.sort(key=lambda e: e.timestamp)

    def get_modalities_present(self) -> Set[ModalityType]:
        return {e.modality for e in self.events}

    def get_event_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {m.value: 0 for m in ModalityType}
        for e in self.events:
            counts[e.modality.value] += 1
        return counts

    def get_date_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        if not self.events:
            return None, None
        return min(e.timestamp for e in self.events), max(e.timestamp for e in self.events)

    def to_dataframe(self) -> pd.DataFrame:
        """Export raw canonical events to a flat DataFrame."""
        records = []
        for e in self.events:
            rec = {
                "patient_id": self.metadata.patient_id,
                "dataset_source": self.metadata.dataset_source,
                "cohort": self.metadata.cohort,
                "split_label": self.metadata.split_label,
                "timestamp": e.timestamp,
                "modality": e.modality.value,
                "value": e.value,
                "value_secondary": e.value_secondary,
                "timestamp_end": e.timestamp_end,
            }
            # Flatten common attributes
            for k, v in e.attributes.items():
                rec[f"attr_{k}"] = v
            records.append(rec)
        return pd.DataFrame(records)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalPatientRecord":
        meta = PatientMetadata.from_dict(data["metadata"])
        events = [CanonicalEvent.from_dict(e) for e in data.get("events", [])]
        return cls(metadata=meta, events=events)

    def save_json(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "CanonicalPatientRecord":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
