"""
OhioT1DM Dataset Adapter.

Translates raw OhioT1DM XML patient files (both 2018 and 2020 cohorts)
into standardized CanonicalPatientRecord structures.
"""

import os
import glob
import logging
import xml.etree.ElementTree as ET
from typing import List, Optional
from datetime import datetime

from module1.adapters.base_adapter import BaseDatasetAdapter
from module1.adapters.registry import AdapterRegistry
from module1.schema.canonical_schema import (
    ModalityType,
    CanonicalEvent,
    PatientMetadata,
    CanonicalPatientRecord,
)

logger = logging.getLogger(__name__)

COHORT_2018_PATIENTS = {"559", "563", "570", "575", "588", "591"}
COHORT_2020_PATIENTS = {"540", "544", "552", "567", "584", "596"}


@AdapterRegistry.register("ohiot1dm")
@AdapterRegistry.register("ohio_t1dm")
class OhioT1DMAdapter(BaseDatasetAdapter):
    """
    Adapter specifically designed for OhioT1DM 2018 & 2020 XML file formats.
    """

    @property
    def dataset_name(self) -> str:
        return "ohiot1dm"

    def discover_files(self, raw_data_dir: str) -> List[str]:
        """
        Recursively discover all OhioT1DM XML files in the target directory.
        """
        pattern = os.path.join(raw_data_dir, "**", "*.xml")
        xml_files = sorted(glob.glob(pattern, recursive=True))
        if not xml_files:
            # Try non-recursive in case directory is flat
            pattern_flat = os.path.join(raw_data_dir, "*.xml")
            xml_files = sorted(glob.glob(pattern_flat))
        logger.info("OhioT1DMAdapter discovered %d XML files in %s", len(xml_files), raw_data_dir)
        return xml_files

    def parse_patient(self, file_path: str, **kwargs) -> CanonicalPatientRecord:
        """
        Parse a single OhioT1DM XML file into a CanonicalPatientRecord.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"OhioT1DM XML file not found: {file_path}")

        tree = ET.parse(file_path)
        root = tree.getroot()

        pid = root.attrib.get("id", "unknown")
        weight_str = root.attrib.get("weight")
        weight_kg = self.safe_float(weight_str)
        insulin_type = root.attrib.get("insulin_type")

        # Determine cohort and split
        norm_path = file_path.replace("\\", "/")
        cohort = "2018" if ("2018" in norm_path or pid in COHORT_2018_PATIENTS) else "2020"
        split_label = "train" if "train" in norm_path.lower() else ("test" if "test" in norm_path.lower() else "unknown")

        metadata = PatientMetadata(
            patient_id=str(pid),
            dataset_source="OhioT1DM",
            cohort=cohort,
            split_label=split_label,
            weight_kg=weight_kg,
            insulin_type=insulin_type,
            extra_info={"file_name": os.path.basename(file_path)},
        )

        record = CanonicalPatientRecord(metadata=metadata)

        for child in root:
            tag = child.tag
            self._parse_xml_section(tag, child, record)

        # Sort all events chronologically
        record.sort_events()
        logger.debug(
            "Parsed OhioT1DM file %s: PID=%s, Total Events=%d",
            os.path.basename(file_path),
            pid,
            len(record.events),
        )
        return record

    def _parse_xml_section(self, tag: str, section_elem: ET.Element, record: CanonicalPatientRecord) -> None:
        """Parse individual XML sections into CanonicalEvents."""
        for ev in section_elem:
            attrib = ev.attrib

            # 1. Glucose level (CGM)
            if tag == "glucose_level":
                ts = self.parse_timestamp(attrib.get("ts"))
                val = self.safe_float(attrib.get("value"))
                if ts and val is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.CGM_GLUCOSE,
                        value=val,
                    ))

            # 2. Finger-stick (BGM)
            elif tag == "finger_stick":
                ts = self.parse_timestamp(attrib.get("ts"))
                val = self.safe_float(attrib.get("value"))
                if ts and val is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.FINGERSTICK_GLUCOSE,
                        value=val,
                    ))

            # 3. Basal (Scheduled Basal Rate)
            elif tag == "basal":
                ts = self.parse_timestamp(attrib.get("ts"))
                val = self.safe_float(attrib.get("value"))
                if ts and val is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.BASAL_RATE,
                        value=val,
                    ))

            # 4. Temp Basal (Temporary Basal Rate Override)
            elif tag == "temp_basal":
                ts_begin = self.parse_timestamp(attrib.get("ts_begin"))
                ts_end = self.parse_timestamp(attrib.get("ts_end"))
                val = self.safe_float(attrib.get("value"))
                if ts_begin and val is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts_begin,
                        modality=ModalityType.TEMP_BASAL_RATE,
                        value=val,
                        timestamp_end=ts_end,
                    ))

            # 5. Bolus (Bolus Insulin Injections)
            elif tag == "bolus":
                ts_begin = self.parse_timestamp(attrib.get("ts_begin"))
                ts_end = self.parse_timestamp(attrib.get("ts_end"))
                dose = self.safe_float(attrib.get("dose"))
                carb_input = self.safe_float(attrib.get("bwz_carb_input"))
                b_type = attrib.get("type", "normal").strip()

                if ts_begin and dose is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts_begin,
                        modality=ModalityType.BOLUS_DOSE,
                        value=dose,
                        value_secondary=carb_input,
                        timestamp_end=ts_end,
                        attributes={"type": b_type, "bwz_carb_input": carb_input},
                    ))

            # 6. Meal (Carbohydrate Intake)
            elif tag == "meal":
                ts = self.parse_timestamp(attrib.get("ts"))
                carbs = self.safe_float(attrib.get("carbs"))
                meal_type = attrib.get("type", "").strip()

                if ts and carbs is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.MEAL_CARBS,
                        value=carbs,
                        attributes={"type": meal_type},
                    ))

            # 7. Wearable: Heart Rate (2018 cohort)
            elif tag == "basis_heart_rate":
                ts = self.parse_timestamp(attrib.get("ts"))
                val = self.safe_float(attrib.get("value"))
                if ts and val is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.HEART_RATE,
                        value=val,
                    ))

            # 8. Wearable: Galvanic Skin Response (GSR)
            elif tag == "basis_gsr":
                ts = self.parse_timestamp(attrib.get("ts"))
                val = self.safe_float(attrib.get("value"))
                if ts and val is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.GSR,
                        value=val,
                    ))

            # 9. Wearable: Skin Temperature
            elif tag == "basis_skin_temperature":
                ts = self.parse_timestamp(attrib.get("ts"))
                val = self.safe_float(attrib.get("value"))
                if ts and val is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.SKIN_TEMPERATURE,
                        value=val,
                    ))

            # 10. Wearable: Air Temperature (2018 cohort)
            elif tag == "basis_air_temperature":
                ts = self.parse_timestamp(attrib.get("ts"))
                val = self.safe_float(attrib.get("value"))
                if ts and val is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.AIR_TEMPERATURE,
                        value=val,
                    ))

            # 11. Wearable: Step count (2018 cohort)
            elif tag == "basis_steps":
                ts = self.parse_timestamp(attrib.get("ts"))
                val = self.safe_float(attrib.get("value"))
                if ts and val is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.STEPS,
                        value=val,
                    ))

            # 12. Wearable: Acceleration (2020 cohort)
            elif tag == "acceleration":
                ts = self.parse_timestamp(attrib.get("ts"))
                val = self.safe_float(attrib.get("value"))
                if ts and val is not None:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.ACCELERATION,
                        value=val,
                    ))

            # 13. Sleep (Self-reported)
            elif tag == "sleep":
                ts_begin = self.parse_timestamp(attrib.get("ts_begin"))
                ts_end = self.parse_timestamp(attrib.get("ts_end"))
                quality = self.safe_float(attrib.get("quality"), default=1.0)
                if ts_begin:
                    record.add_event(CanonicalEvent(
                        timestamp=ts_begin,
                        modality=ModalityType.SLEEP,
                        value=quality if quality is not None else 1.0,
                        timestamp_end=ts_end,
                        attributes={"source": "self_reported", "quality": quality},
                    ))

            # 14. Basis Sleep (Wristband detected)
            elif tag == "basis_sleep":
                tbegin = self.parse_timestamp(attrib.get("tbegin"))
                tend = self.parse_timestamp(attrib.get("tend"))
                quality = self.safe_float(attrib.get("quality"), default=1.0)
                slp_type = attrib.get("type", "").strip()
                if tbegin:
                    record.add_event(CanonicalEvent(
                        timestamp=tbegin,
                        modality=ModalityType.SLEEP,
                        value=quality if quality is not None else 1.0,
                        timestamp_end=tend,
                        attributes={"source": "basis_band", "type": slp_type},
                    ))

            # 15. Exercise
            elif tag == "exercise":
                ts = self.parse_timestamp(attrib.get("ts"))
                intensity = self.safe_float(attrib.get("intensity"), default=1.0)
                duration = self.safe_float(attrib.get("duration"))
                ex_type = attrib.get("type", "").strip()
                if ts:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.EXERCISE,
                        value=intensity if intensity is not None else 1.0,
                        value_secondary=duration,
                        attributes={"type": ex_type, "duration": duration},
                    ))

            # 16. Work
            elif tag == "work":
                ts_begin = self.parse_timestamp(attrib.get("ts_begin"))
                ts_end = self.parse_timestamp(attrib.get("ts_end"))
                intensity = self.safe_float(attrib.get("intensity"), default=1.0)
                if ts_begin:
                    record.add_event(CanonicalEvent(
                        timestamp=ts_begin,
                        modality=ModalityType.WORK,
                        value=intensity if intensity is not None else 1.0,
                        timestamp_end=ts_end,
                    ))

            # 17. Illness
            elif tag == "illness":
                ts_begin = self.parse_timestamp(attrib.get("ts_begin"))
                ts_end = self.parse_timestamp(attrib.get("ts_end"))
                desc = attrib.get("description", "").strip()
                if ts_begin:
                    record.add_event(CanonicalEvent(
                        timestamp=ts_begin,
                        modality=ModalityType.ILLNESS,
                        value=1.0,
                        timestamp_end=ts_end,
                        attributes={"description": desc},
                    ))

            # 18. Stressors
            elif tag == "stressors":
                ts = self.parse_timestamp(attrib.get("ts"))
                desc = attrib.get("description", "").strip()
                if ts:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.STRESS,
                        value=1.0,
                        attributes={"description": desc},
                    ))

            # 19. Hypo Event
            elif tag == "hypo_event":
                ts = self.parse_timestamp(attrib.get("ts"))
                if ts:
                    record.add_event(CanonicalEvent(
                        timestamp=ts,
                        modality=ModalityType.HYPO_EVENT,
                        value=1.0,
                    ))
