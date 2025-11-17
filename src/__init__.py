"""
Slotless Math Model Package

This package implements electromagnetic field calculations for slotless motors 
based on Zhu's mathematical models.

Modules:
- open_circuit_field: Open-circuit magnetic field calculations (Zhu Part 1)
- armature_reaction_field: Armature reaction field calculations (Zhu Part 2)
- stator_slotting_effect: Effect of stator slotting (Zhu Part 3)
- magnetic_field_on_load: Magnetic field under load conditions (Zhu Part 4)
"""

__version__ = "1.0.0"
__author__ = "Electromagnetic Calculation Team"

from .open_circuit_field import OpenCircuitField
from .armature_reaction_field import ArmatureReactionField
from .stator_slotting_effect import StatorSlottingEffect
from .magnetic_field_on_load import MagneticFieldOnLoad

__all__ = [
    'OpenCircuitField',
    'ArmatureReactionField', 
    'StatorSlottingEffect',
    'MagneticFieldOnLoad'
]