"""
Adapters package for dataset ingestion and canonicalization.
"""

from module1.adapters.base_adapter import BaseDatasetAdapter
from module1.adapters.registry import AdapterRegistry
from module1.adapters.ohio_t1dm_adapter import OhioT1DMAdapter
from module1.adapters.hupa_ucm_adapter import HUPAUCMAdapter

__all__ = [
    "BaseDatasetAdapter",
    "AdapterRegistry",
    "OhioT1DMAdapter",
    "HUPAUCMAdapter",
]
