"""
Adapter Registry for Dynamic Dataset Discovery and Plug-and-Play Extensibility.
"""

from typing import Dict, Type, Optional, List
import logging
from module1.adapters.base_adapter import BaseDatasetAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Central registry mapping dataset identifier names to adapter classes."""
    _registry: Dict[str, Type[BaseDatasetAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an adapter class."""
        def decorator(subclass: Type[BaseDatasetAdapter]):
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get_adapter(cls, name: str) -> BaseDatasetAdapter:
        """Instantiate and return the registered adapter for the dataset."""
        key = name.lower()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown dataset adapter '{name}'. Available adapters: {available}")
        return cls._registry[key]()

    @classmethod
    def list_adapters(cls) -> List[str]:
        """List all registered dataset adapter names."""
        return sorted(list(cls._registry.keys()))
