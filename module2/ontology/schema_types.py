"""
Type definitions, enums, and constants for ontology schema mapping.
"""

from enum import IntEnum, unique
from typing import Dict, List, Set, Tuple, Any

# Ontology Namespace
DEFAULT_ONTOLOGY_URI = "http://example.org/diabetes-ontology#"


@unique
class NodeType(IntEnum):
    """
    Unique integer indices for each ontology node category.
    Used to generate one-hot node type encodings in feature matrix X.
    """
    PATIENT = 0
    TIME_WINDOW = 1
    GLUCOSE_OBSERVATION = 2
    MEAL_OBSERVATION = 3
    BASAL_INSULIN = 4
    TEMP_BASAL_INSULIN = 5
    BOLUS_INSULIN = 6  # Historical boluses strictly before prediction time t
    HEART_RATE_OBSERVATION = 7
    SKIN_TEMP_OBSERVATION = 8
    AIR_TEMP_OBSERVATION = 9
    GSR_OBSERVATION = 10
    ACCELERATION_OBSERVATION = 11
    STEP_OBSERVATION = 12


TOTAL_NODE_TYPES: int = len(NodeType)  # 13 node types -> 13-dim one-hot vector


@unique
class EdgeType(IntEnum):
    """
    Unique integer indices for edge categories in heterogeneous/typed message passing.
    """
    HAS_TIME_WINDOW = 0       # Patient -> TimeWindow
    HAS_OBSERVATION = 1       # TimeWindow -> Observation
    SEQUENTIAL_TEMPORAL = 2   # Modality_(t_i) -> Modality_(t_i+1) (sparse consecutive)
    CONTEMPORANEOUS = 3       # Cross-modal co-occurrence (e.g. Meal <-> Glucose at same step)


# Mapping from Module 1 grid column names to corresponding NodeType
COLUMN_TO_NODE_TYPE: Dict[str, NodeType] = {
    "cgm_glucose": NodeType.GLUCOSE_OBSERVATION,
    "fingerstick_glucose": NodeType.GLUCOSE_OBSERVATION,
    "meal_carbs": NodeType.MEAL_OBSERVATION,
    "basal_rate": NodeType.BASAL_INSULIN,
    "temp_basal_rate": NodeType.TEMP_BASAL_INSULIN,
    "bolus_dose": NodeType.BOLUS_INSULIN,
    "heart_rate": NodeType.HEART_RATE_OBSERVATION,
    "skin_temperature": NodeType.SKIN_TEMP_OBSERVATION,
    "air_temperature": NodeType.AIR_TEMP_OBSERVATION,
    "gsr": NodeType.GSR_OBSERVATION,
    "acceleration": NodeType.ACCELERATION_OBSERVATION,
    "steps": NodeType.STEP_OBSERVATION,
}

# Mapping from NodeType to the exact OWL class name in diabetes_insulin_ontology.owl
NODE_TYPE_TO_OWL_CLASS: Dict[NodeType, str] = {
    NodeType.PATIENT: "Patient",
    NodeType.TIME_WINDOW: "TimeWindow",
    NodeType.GLUCOSE_OBSERVATION: "GlucoseObservation",
    NodeType.MEAL_OBSERVATION: "MealObservation",
    NodeType.BASAL_INSULIN: "BasalInsulinAdministration",
    NodeType.TEMP_BASAL_INSULIN: "TempBasalInsulinAdministration",
    NodeType.BOLUS_INSULIN: "BolusInsulinAdministration",
    NodeType.HEART_RATE_OBSERVATION: "HeartRateObservation",
    NodeType.SKIN_TEMP_OBSERVATION: "SkinTemperatureObservation",
    NodeType.AIR_TEMP_OBSERVATION: "AirTemperatureObservation",
    NodeType.GSR_OBSERVATION: "GalvanicSkinResponseObservation",
    NodeType.ACCELERATION_OBSERVATION: "AccelerationObservation",
    NodeType.STEP_OBSERVATION: "StepObservation",
}

# Expected OWL Classes in the ontology
ONTOLOGY_CLASSES: Set[str] = {
    "Patient",
    "TimeWindow",
    "Observation",
    "VitalSignObservation",
    "GlucoseObservation",
    "MealObservation",
    "SleepObservation",
    "ExerciseObservation",
    "HeartRateObservation",
    "SkinTemperatureObservation",
    "AirTemperatureObservation",
    "GalvanicSkinResponseObservation",
    "AccelerationObservation",
    "StepObservation",
    "InsulinAdministration",
    "BasalInsulinAdministration",
    "TempBasalInsulinAdministration",
    "BolusInsulinAdministration",
}

# Expected Properties in the ontology
ONTOLOGY_PROPERTIES: Set[str] = {
    "hasTimeWindow",
    "hasObservation",
    "temporallyPrecedes",
    "patientId",
    "timestamp",
    "windowId",
    "windowStart",
    "windowEnd",
    "glucoseValue",
    "glucoseSourceType",
    "carbohydrateAmount",
    "heartRateValue",
    "skinTemperatureValue",
    "airTemperatureValue",
    "skinConductanceValue",
    "accelerationValue",
    "stepCount",
    "basalRate",
    "insulinDose",
}
