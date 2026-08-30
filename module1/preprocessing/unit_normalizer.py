"""
Dataset-Agnostic Unit Normalization Engine.

Standardizes physiological, nutritional, and pharmacological variables into
the project's canonical units with explicit audit logging of every conversion.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import logging

from module1.schema.canonical_schema import (
    CanonicalPatientRecord,
    CanonicalEvent,
    ModalityType,
    CANONICAL_UNITS,
)

logger = logging.getLogger(__name__)


@dataclass
class UnitConversionEntry:
    modality: str
    from_unit: str
    to_unit: str
    conversion_formula: str
    records_converted: int


@dataclass
class UnitConversionLog:
    patient_id: str
    entries: List[UnitConversionEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "conversions": [
                {
                    "modality": e.modality,
                    "from_unit": e.from_unit,
                    "to_unit": e.to_unit,
                    "conversion_formula": e.conversion_formula,
                    "records_converted": e.records_converted,
                }
                for e in self.entries
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            f"### Unit Normalization Log: Patient {self.patient_id}",
            "",
        ]
        if self.entries:
            lines.append("| Modality | Source Unit | Canonical Target Unit | Formula | Records Converted |")
            lines.append("|---|---|---|---|---|")
            for e in self.entries:
                lines.append(f"| {e.modality} | {e.from_unit} | {e.to_unit} | `{e.conversion_formula}` | {e.records_converted:,} |")
        else:
            lines.append("✓ All records already conformed to project canonical units (no conversion required).")
        return "\n".join(lines)


class UnitNormalizer:
    """
    Normalizes multi-modal event streams into standard units.
    """

    def __init__(self, target_temperature_unit: str = "degF"):
        self.target_temp_unit = target_temperature_unit  # "degF" or "degC"

    def normalize(
        self,
        record: CanonicalPatientRecord,
        source_glucose_unit: str = "mg/dL",
        source_temperature_unit: str = "degF",
    ) -> Tuple[CanonicalPatientRecord, UnitConversionLog]:
        """
        Normalize units of all events in the CanonicalPatientRecord.
        """
        log = UnitConversionLog(patient_id=record.metadata.patient_id)
        normalized_events: List[CanonicalEvent] = []

        glucose_conversions = 0
        temp_conversions = 0

        # Convert mmol/L to mg/dL if needed
        need_glucose_conv = source_glucose_unit.lower() in ("mmol/l", "mmol")
        # Convert temperature if target differs from source
        need_temp_conv = (source_temperature_unit.lower() != self.target_temp_unit.lower())

        for event in record.events:
            val = event.value
            sec_val = event.value_secondary

            # Glucose conversion
            if event.modality in (ModalityType.CGM_GLUCOSE, ModalityType.FINGERSTICK_GLUCOSE):
                if need_glucose_conv:
                    val = val * 18.0182
                    glucose_conversions += 1

            # Temperature conversion
            elif event.modality in (ModalityType.SKIN_TEMPERATURE, ModalityType.AIR_TEMPERATURE):
                if need_temp_conv:
                    if source_temperature_unit.lower() == "degf" and self.target_temp_unit.lower() == "degc":
                        val = (val - 32.0) * (5.0 / 9.0)
                        temp_conversions += 1
                    elif source_temperature_unit.lower() == "degc" and self.target_temp_unit.lower() == "degf":
                        val = (val * 9.0 / 5.0) + 32.0
                        temp_conversions += 1

            normalized_events.append(CanonicalEvent(
                timestamp=event.timestamp,
                modality=event.modality,
                value=val,
                value_secondary=sec_val,
                timestamp_end=event.timestamp_end,
                attributes=event.attributes.copy(),
            ))

        if glucose_conversions > 0:
            log.entries.append(UnitConversionEntry(
                modality="glucose",
                from_unit=source_glucose_unit,
                to_unit="mg/dL",
                conversion_formula="mg/dL = mmol/L * 18.0182",
                records_converted=glucose_conversions,
            ))

        if temp_conversions > 0:
            log.entries.append(UnitConversionEntry(
                modality="temperature",
                from_unit=source_temperature_unit,
                to_unit=self.target_temp_unit,
                conversion_formula=f"{self.target_temp_unit} standard conversion",
                records_converted=temp_conversions,
            ))

        normalized_record = CanonicalPatientRecord(
            metadata=record.metadata,
            events=normalized_events,
        )

        return normalized_record, log
