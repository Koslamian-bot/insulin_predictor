"""
Base Dataset Adapter Interface and Contract.

Defines the abstract interface that all dataset-specific adapters must implement
to translate raw proprietary dataset formats into the unified Canonical Schema.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import logging

from module1.schema.canonical_schema import CanonicalPatientRecord

logger = logging.getLogger(__name__)


class BaseDatasetAdapter(ABC):
    """
    Abstract Base Adapter.
    Any dataset (OhioT1DM, HUPA-UCM, future datasets) implements this interface.
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the unique identifier for this dataset."""
        pass

    @abstractmethod
    def discover_files(self, raw_data_dir: str) -> List[str]:
        """
        Discover all raw data files in the given directory for this dataset.
        
        Args:
            raw_data_dir: Path to directory containing raw dataset files.
            
        Returns:
            List of absolute paths to individual patient files.
        """
        pass

    @abstractmethod
    def parse_patient(self, file_path: str, **kwargs) -> CanonicalPatientRecord:
        """
        Parse an individual patient file into a CanonicalPatientRecord.
        
        Args:
            file_path: Path to the raw patient file.
            
        Returns:
            CanonicalPatientRecord containing standardized events and metadata.
        """
        pass

    def load_all(self, raw_data_dir: str, **kwargs) -> List[CanonicalPatientRecord]:
        """
        Discover and parse all patient files in the raw dataset directory.
        
        Args:
            raw_data_dir: Path to directory containing raw dataset files.
            
        Returns:
            List of CanonicalPatientRecord objects.
        """
        files = self.discover_files(raw_data_dir)
        logger.info("Found %d patient files for dataset '%s' in %s", len(files), self.dataset_name, raw_data_dir)
        records: List[CanonicalPatientRecord] = []
        for file_path in files:
            rec = self.parse_patient(file_path, **kwargs)
            records.append(rec)
        return records

    @staticmethod
    def parse_timestamp(
        ts_str: Optional[str],
        formats: Optional[List[str]] = None
    ) -> Optional[datetime]:
        """
        Utility to robustly parse timestamp strings across formats.
        """
        if not ts_str or not ts_str.strip():
            return None
        
        s = ts_str.strip()
        default_formats = [
            "%d-%m-%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S.%f",
            "%d/%m/%Y %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d-%m-%Y %H:%M",
            "%Y-%m-%d %H:%M",
        ]
        active_formats = formats or default_formats

        for fmt in active_formats:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue

        # Try ISO format as fallback
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            logger.warning("Could not parse timestamp: '%s'", ts_str)
            return None

    @staticmethod
    def safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
        """Utility to safely cast values to float."""
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
