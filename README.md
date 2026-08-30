# Diabetes Multi-Modal Insulin Prediction Engine — Module 1

## Module 1: Reusable, Dataset-Agnostic Data Ingestion & Preprocessing Engine

Module 1 is the foundational data pipeline of the Insulin Prediction project. It ingests raw multi-modal time-series diabetes data (CGM glucose, insulin pumps, meals, and wearable biometric sensors), canonicalizes heterogeneous schemas into a unified standard, and executes rigorous, leakage-safe preprocessing to produce model-ready training, validation, and test datasets.

---

## 1. Core Architecture

The system is decoupled into two independent layers:

```
RAW DATASET (OhioT1DM / HUPA-UCM / Future Dataset)
      ↓
[LAYER A] DATASET ADAPTER (OhioT1DMAdapter / HUPAUCMAdapter)
      ↓
[CANONICAL SCHEMA] Standardized Multi-Modal Events & Metadata
      ↓
[LAYER B] GENERIC PREPROCESSING ENGINE
      ├── 1. Data Validation (Range, ordering, completeness checks)
      ├── 2. Data Cleaning & Audit Trail (Deduplication, corruption filtering)
      ├── 3. Unit Normalization (mg/dL, Units, grams, BPM, degF/degC)
      ├── 4. Timestamp Synchronization (5-minute uniform grid, discrete conservation)
      ├── 5. Missing-Data Handling (Short-gap interpolation <= 15m, explicit masks)
      ├── 6. Causal Feature Engineering (ROC, rolling statistics, IOB decay, cyclics)
      ├── 7. Insulin Target Generation (Ground-truth bolus delivery & future horizons)
      ├── 8. Leakage-Safe Splitting & Scaling (Train-only fitted normalization)
      └── 9. Quality Reporting (Modality matrices & markdown quality reports)
      ↓
PROCESSED CANONICAL DATASET
      ↓
Module 2: Ontology / Knowledge Graph & Deep Learning Models (LSTM/BiLSTM/GCN)
```

---

## 2. File-by-File Breakdown

### `module1/schema/`
- **`canonical_schema.py`**:
  - `ModalityType`: Enum of all 18 modalities (CGM, BGM, Basal, Bolus, Meals, HR, GSR, Skin Temp, Accel, Steps, Sleep, Exercise, etc.).
  - `CanonicalEvent`: Atomic event container (`timestamp`, `modality`, `value`, `value_secondary`, `attributes`).
  - `PatientMetadata`: Provenance tracking (`patient_id`, `dataset_source`, `cohort`, `split_label`, `weight_kg`, `insulin_type`).
  - `CanonicalPatientRecord`: Chronological collection of canonical events with export to flat DataFrames and JSON.
  - `CANONICAL_UNITS` & `PHYSIOLOGICAL_BOUNDS`: Standard project units and strict clinical bounds.

### `module1/adapters/`
- **`base_adapter.py`**: Abstract base class (`BaseDatasetAdapter`) defining discovery, parsing, and timestamp parsing utilities.
- **`registry.py`**: Dynamic plug-and-play adapter registry (`AdapterRegistry`) allowing datasets to be registered via `@AdapterRegistry.register("name")`.
- **`ohio_t1dm_adapter.py`**: Specialized parser for OhioT1DM 2018 & 2020 cohorts XML files.
- **`hupa_ucm_adapter.py`**: Specialized parser for HUPA-UCM CSV datasets (supporting semicolon and comma formats).

### `module1/validation/`
- **`validator.py`**: Evaluates metadata, chronological ordering, duplicate events, physiological bounds, and critical modality presence.
- **`validation_report.py`**: Formats structured validation outcomes into `PASS`, `WARNING`, or `ERROR` severities.

### `module1/preprocessing/`
- **`cleaner.py`**: Deduplicates records, discards physiologically impossible entries, and maintains an immutable `CleaningAuditTrail`.
- **`unit_normalizer.py`**: Standardizes units (e.g., mmol/L to mg/dL, temperature conversions) with explicit `UnitConversionLog`.
- **`timestamp_sync.py`**: High-performance vectorized synchronizer that aligns continuous and discrete multi-modal streams to a uniform 5-minute grid while strictly conserving bolus insulin units and carbohydrate grams.
- **`missing_data.py`**: Distinguishes discrete absence (0 events) vs continuous sensor dropouts (NaN), interpolates short physiological gaps ($\le 15\text{ min}$), and creates explicit binary masks (`cgm_missing_mask`, `cgm_interpolated`, and modality availability indicators).
- **`feature_engineering.py`**: Computes strictly backward-looking (causal) features:
  - Glucose rates of change (5m, 15m) and acceleration.
  - Rolling mean, std, min, max, range, and EWMA across 30m, 60m, 120m.
  - Meal intake history (carbs in 30m, 60m, 120m, 240m) and time elapsed since last meal.
  - Insulin history (bolus in 30m, 60m, 120m, 240m) and time elapsed since last bolus.
  - **Pharmacokinetic Insulin-On-Board (IOB)** using bilinear exponential decay ($DIA = 3.5\text{ hours}$).
  - Circadian sinusoidal features ($\sin/\cos$ hour of day, day of week, weekend indicator).
  - Wearable sensor rolling statistics (HR, GSR, Skin Temp, Acceleration, Steps).
- **`target_builder.py`**: Generates genuine clinical insulin targets without synthetic fabrication:
  - `target_current_bolus_dose`: Actual bolus dose delivered at this step.
  - `target_has_bolus`: Binary bolus delivery event flag.
  - `target_next_step_total_insulin`: Total insulin requirement at step $t+1$.
  - `target_future_bolus_30m` / `60m`: Forward-looking rolling bolus sums over 30 and 60 minutes.
- **`splitter_scaler.py`**: Partitions data chronologically and patient-wise. Scalers (`StandardScaler`, `RobustScaler`, `MinMaxScaler`) are fit **only on training data** and transformed onto validation and test sets.

### `module1/reports/`
- **`quality_report.py`**: Aggregates dataset metrics into a comprehensive Data Quality Markdown Report, Modality Availability Matrix, and JSON summaries.

### `module1/pipeline/`
- **`preprocessing_pipeline.py`**: Top-level orchestrator connecting all components end-to-end.

---

## 3. How to Run Module 1

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Option A: Using the CLI Entry Point
Run the full OhioT1DM pipeline with default settings:
```bash
python run_pipeline.py
```

List all registered dataset adapters:
```bash
python run_pipeline.py --list-adapters
```

Run HUPA-UCM preprocessing:
```bash
python run_pipeline.py --dataset hupa_ucm --input "HUPA-UCM Diabetes Dataset/Preprocessed" --output data_hupa
```

Custom parameters:
```bash
python run_pipeline.py --dataset ohiot1dm --input OhioT1DM --output data --config module1/config/preprocessing_config.yaml
```

### Option B: Using the Python API
```python
from module1.pipeline import preprocess

results = preprocess(
    dataset="ohiot1dm",
    input_path="OhioT1DM",
    output_path="data"
)
print("Processed rows:", results["total_grid_rows"])
```

### Option C: Running the Full Test Suite
Execute the unit and integration tests:
```bash
python -m unittest discover tests
```

---

## 4. Outcomes and Output Artifacts

Running the pipeline creates the following directory structure:

```
data/
├── canonical/
│   ├── patient_559_train_2018.json
│   ├── patient_540_train_2020.json
│   └── ... (24 canonical JSON patient records)
├── processed/
│   ├── canonical_timeseries_all.csv    # Full unified grid (188,980 rows x 88 features)
│   ├── canonical_train.csv             # Scaled training fold (130,104 rows)
│   ├── canonical_val.csv               # Scaled validation fold (22,951 rows)
│   └── canonical_test.csv              # Scaled testing fold (35,925 rows)
├── reports/
│   ├── data_quality_report.md          # Full dataset quality summary
│   ├── modality_availability.md        # Patient-by-patient modality matrix
│   ├── quality_report.json             # Structured JSON metrics
│   └── cleaning_audit_trail.json       # Record of all filtered events
└── scalers/
    └── scaler_params.json              # Trained mean/scale normalization parameters
```

---

## 5. Adding Future Datasets

To support any new diabetes dataset (e.g. Cambridge, UVA/Padova, DiaTrend):
1. Subclass `BaseDatasetAdapter` in `module1/adapters/`
2. Register it: `@AdapterRegistry.register("dataset_name")`
3. Implement `discover_files()` and `parse_patient()` mapping to `CanonicalEvent`
4. Run `python run_pipeline.py --dataset dataset_name --input path/to/dataset`
