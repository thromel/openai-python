"""
LLM Design by Contract Framework

A comprehensive Design by Contract framework for Large Language Model APIs
that provides input/output validation, temporal contracts, streaming support,
and multi-platform compatibility.
"""

__version__ = "0.1.0"
__author__ = "Romel"

from .core.interfaces import ContractBase, ValidatorBase, ProviderAdapter
from .core.exceptions import ContractViolationError, ValidationError, ProviderError

# Import submodules for easier access
from . import providers
from . import validators
from . import contracts

__all__ = [
    "ContractBase",
    "ValidatorBase",
    "ProviderAdapter",
    "ContractViolationError",
    "ValidationError",
    "ProviderError",
    "providers",
    "validators",
    "contracts",
]
