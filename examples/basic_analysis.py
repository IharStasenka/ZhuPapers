"""
Example: Basic Motor Analysis

This example demonstrates basic electromagnetic analysis of a slotless PM motor
using individual modules from the Zhu mathematical model implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from open_circuit_field import MotorGeometry, MagnetProperties, OpenCircuitField
from armature_reaction_field import WindingConfiguration, ArmatureReactionField


def basic_motor_example():
    """Demonstrate basic motor analysis workflow."""
    
    print("Basic Slotless Motor Analysis Example")
    print("=" * 50)
    
    # Define a small servo motor
    geometry = MotorGeometry(
        stator_inner_radius=0.025,   # 25 mm
        stator_outer_radius=0.040,   # 40 mm
        rotor_outer_radius=0.023,    # 23 mm
        rotor_inner_radius=0.008,    # 8 mm
        airgap_length=0.002,         # 2 mm
        axial_length=0.050,          # 50 mm
        pole_pairs=2                 # 2 pole pairs (4 poles)
    )
    
    # NdFeB magnet properties
    magnet = MagnetProperties(
        residual_flux_density=1.1,   # 1.1 T
        relative_permeability=1.05,
        coercivity=850000,           # 850 kA/m
        magnet_thickness=0.003,      # 3 mm
        magnet_arc_ratio=0.75        # 75% of pole pitch
    )
    
    # Simple concentrated winding
    winding = WindingConfiguration(
        phases=3,
        slots_per_pole_per_phase=1.0,
        turns_per_coil=30,
        coil_span=1.0,               # Full pitch
        current_density=4e6,         # 4 A/mm²
        wire_width=1e-3,
        wire_height=2e-3,
    )
    
    # 1. Open-circuit analysis
    print("\\n1. Open-Circuit Field Analysis")
    print("-" * 30)
    
    oc_field = OpenCircuitField(geometry, magnet)
    
    # Calculate flux linkage
    flux_linkage = oc_field.flux_linkage_per_turn(geometry.stator_inner_radius)
    print(f"PM flux linkage per turn: {flux_linkage*1000:.3f} mWb")
    
    # Harmonic analysis
    harmonics, amplitudes = oc_field.harmonic_analysis(n_harmonics=20)
    fundamental_idx = geometry.pole_pairs - 1
    if fundamental_idx < len(amplitudes):
        fundamental = amplitudes[fundamental_idx]
        print(f"Fundamental flux density: {fundamental:.3f} T")
        
        # Calculate THD
        harmonics_sum = np.sum(amplitudes[fundamental_idx+1:fundamental_idx+10]**2)
        thd = np.sqrt(harmonics_sum) / fundamental * 100
        print(f"THD (first 10 harmonics): {thd:.2f}%")
    
    # 2. Armature reaction analysis  
    print("\\n2. Armature Reaction Analysis")
    print("-" * 30)
    
    ar_field = ArmatureReactionField(geometry, winding)
    
    # Calculate inductances
    inductances = ar_field.armature_reaction_inductance()
    print(f"Self inductance: {inductances['self_inductance']*1000:.2f} mH")
    print(f"Mutual inductance: {inductances['mutual_inductance']*1000:.2f} mH")
    
    # 3. Generate plots
    print("\\n3. Generating Plots")
    print("-" * 30)
    
    # Plot open-circuit field
    plt.figure(figsize=(12, 8))
    oc_field.plot_flux_density_distribution()
    plt.suptitle('Open-Circuit Magnetic Field Distribution')
    
    # Plot armature reaction field
    plt.figure(figsize=(12, 8))
    ar_field.plot_armature_flux_density(current_magnitude=20.0)
    plt.suptitle('Armature Reaction Field at 20A')
    
    # Show plots
    plt.show()
    
    print("\\nBasic analysis complete!")


if __name__ == "__main__":
    basic_motor_example()