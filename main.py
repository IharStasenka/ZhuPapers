"""
Main Integration Module for Slotless Math Model

This module demonstrates the complete electromagnetic analysis workflow
based on Zhu's papers, integrating all four parts:
1. Open-circuit field calculation
2. Armature reaction field 
3. Effect of stator slotting
4. Magnetic field on load

Author: Electromagnetic Calculation Team
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import time
import os

# Import all modules
from src.open_circuit_field import MotorGeometry, MagnetProperties, OpenCircuitField
from src.armature_reaction_field import WindingConfiguration, ArmatureReactionField
from src.stator_slotting_effect import SlotGeometry, StatorSlottingEffect
from src.magnetic_field_on_load import LoadConditions, MagneticFieldOnLoad


class SlotlessMotorAnalyzer:
    """
    Complete electromagnetic analysis tool for slotless PM motors.
    
    Integrates all four parts of Zhu's mathematical model for comprehensive
    motor design and performance analysis.
    """
    
    def __init__(self, geometry: MotorGeometry, magnet: MagnetProperties,
                 winding: WindingConfiguration, slot_geometry: Optional[SlotGeometry] = None):
        """
        Initialize the complete motor analyzer.
        
        Args:
            geometry: Motor geometric parameters
            magnet: Permanent magnet properties
            winding: Stator winding configuration
            slot_geometry: Slot geometry (optional, for slotted motors)
        """
        self.geometry = geometry
        self.magnet = magnet
        self.winding = winding
        self.slot_geometry = slot_geometry
        
        # Initialize all analysis modules
        print("Initializing electromagnetic analysis modules...")
        self.oc_field = OpenCircuitField(geometry, magnet)
        self.ar_field = ArmatureReactionField(geometry, winding)
        
        if slot_geometry:
            self.slotting = StatorSlottingEffect(geometry, slot_geometry)
            self.load_field = MagneticFieldOnLoad(geometry, magnet, winding, slot_geometry)
        else:
            self.slotting = None
            self.load_field = MagneticFieldOnLoad(geometry, magnet, winding)
        
        print("Initialization complete!")
    
    def complete_analysis(self, load_conditions: LoadConditions, 
                         save_results: bool = True,
                         output_dir: str = "results") -> Dict:
        """
        Perform complete electromagnetic analysis.
        
        Args:
            load_conditions: Operating load conditions
            save_results: Whether to save plots and results
            output_dir: Directory to save results
            
        Returns:
            Dictionary with all analysis results
        """
        print("\\n" + "="*60)
        print("COMPLETE ELECTROMAGNETIC ANALYSIS")
        print("Based on Zhu's Mathematical Model")
        print("="*60)
        
        results = {}
        
        if save_results:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # 1. Open-Circuit Field Analysis (Zhu Part 1)
        print("\\n1. OPEN-CIRCUIT FIELD ANALYSIS (Zhu Part 1)")
        print("-" * 50)
        
        # Calculate PM flux linkage
        pm_flux_linkage = self.oc_field.flux_linkage_per_turn(self.geometry.stator_inner_radius)
        print(f"PM flux linkage per turn: {pm_flux_linkage*1000:.2f} mWb")
        
        # Harmonic analysis
        harmonics, amplitudes = self.oc_field.harmonic_analysis()
        fundamental_amplitude = amplitudes[self.geometry.pole_pairs - 1] if len(amplitudes) >= self.geometry.pole_pairs else 0
        print(f"Fundamental flux density amplitude: {fundamental_amplitude:.3f} T")
        
        # Calculate THD
        thd = np.sqrt(np.sum(amplitudes[self.geometry.pole_pairs:]**2)) / fundamental_amplitude * 100
        print(f"Total Harmonic Distortion (THD): {thd:.2f}%")
        
        results['open_circuit'] = {
            'pm_flux_linkage': pm_flux_linkage,
            'fundamental_amplitude': fundamental_amplitude,
            'thd': thd,
            'harmonics': harmonics,
            'amplitudes': amplitudes
        }
        
        # 2. Armature Reaction Analysis (Zhu Part 2)
        print("\\n2. ARMATURE REACTION ANALYSIS (Zhu Part 2)")
        print("-" * 50)
        
        # Calculate inductances
        inductances = self.ar_field.armature_reaction_inductance()
        print(f"Self inductance: {inductances['self_inductance']*1000:.2f} mH")
        print(f"Mutual inductance: {inductances['mutual_inductance']*1000:.2f} mH")
        print(f"Cyclic inductance: {inductances['cyclic_inductance']*1000:.2f} mH")
        
        results['armature_reaction'] = inductances
        
        # 3. Slotting Effect Analysis (Zhu Part 3)
        if self.slotting:
            print("\\n3. STATOR SLOTTING EFFECT ANALYSIS (Zhu Part 3)")
            print("-" * 50)
            
            # Calculate Carter's coefficient
            carter_coeff = self.slotting.carter_coefficient
            print(f"Carter's coefficient: {carter_coeff:.3f}")
            
            # Slot leakage inductance
            slot_inductance = self.slotting.calculate_slot_leakage_inductance()
            print(f"Slot leakage inductance: {slot_inductance*1000:.2f} mH")
            
            # Slot harmonic analysis
            slot_harmonics, slot_amplitudes = self.slotting.slot_harmonic_analysis()
            print(f"Dominant slot harmonic order: {slot_harmonics[np.argmax(slot_amplitudes)]}")
            print(f"Dominant slot harmonic amplitude: {np.max(slot_amplitudes):.4f}")
            
            results['slotting'] = {
                'carter_coefficient': carter_coeff,
                'slot_inductance': slot_inductance,
                'slot_harmonics': slot_harmonics,
                'slot_amplitudes': slot_amplitudes
            }
        
        # 4. Magnetic Field on Load Analysis (Zhu Part 4)
        print("\\n4. MAGNETIC FIELD ON LOAD ANALYSIS (Zhu Part 4)")
        print("-" * 50)
        
        # Electromagnetic torque
        em_torque = self.load_field.electromagnetic_torque(load_conditions)
        print(f"Electromagnetic torque: {em_torque:.2f} Nm")
        
        # Motor performance
        performance = self.load_field.motor_performance_analysis(load_conditions)
        print(f"Efficiency: {performance.efficiency*100:.1f}%")
        print(f"Input power: {performance.input_power:.1f} W")
        print(f"Output power: {performance.output_power:.1f} W")
        print(f"Copper losses: {performance.copper_losses:.1f} W")
        print(f"Iron losses: {performance.iron_losses:.1f} W")
        
        # Iron loss breakdown
        iron_losses = self.load_field.iron_loss_calculation(load_conditions)
        print(f"Hysteresis losses: {iron_losses['hysteresis_loss']:.1f} W")
        print(f"Eddy current losses: {iron_losses['eddy_current_loss']:.1f} W")
        
        results['load_analysis'] = {
            'electromagnetic_torque': em_torque,
            'performance': performance,
            'iron_losses': iron_losses
        }
        
        # 5. Generate comprehensive plots
        if save_results:
            print("\\n5. GENERATING ANALYSIS PLOTS")
            print("-" * 50)
            self._generate_all_plots(load_conditions, output_dir)
            print(f"Plots saved to '{output_dir}' directory")
        
        print("\\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return results
    
    def _generate_all_plots(self, load_conditions: LoadConditions, output_dir: str):
        """Generate all analysis plots and save them."""
        
        # Set matplotlib to non-interactive mode for saving
        plt.ioff()
        
        # 1. Open-circuit field distribution
        plt.figure(figsize=(12, 8))
        self.oc_field.plot_flux_density_distribution()
        plt.savefig(f"{output_dir}/01_open_circuit_field.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Armature reaction field
        plt.figure(figsize=(12, 8))
        self.ar_field.plot_armature_flux_density(current_magnitude=load_conditions.rms_current)
        plt.savefig(f"{output_dir}/02_armature_reaction_field.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Current sheet distribution
        plt.figure(figsize=(12, 8))
        self.ar_field.plot_current_sheet_distribution(current_magnitude=load_conditions.rms_current)
        plt.savefig(f"{output_dir}/03_current_sheet_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Slotting effect (if applicable)
        if self.slotting:
            plt.figure(figsize=(12, 6))
            self.slotting.plot_permeance_function()
            plt.savefig(f"{output_dir}/04_permeance_function.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(12, 6))
            self.slotting.plot_cogging_torque()
            plt.savefig(f"{output_dir}/05_cogging_torque.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Complete field on load
        plt.figure(figsize=(15, 10))
        self.load_field.plot_load_field_distribution(load_conditions)
        plt.savefig(f"{output_dir}/06_field_on_load.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Flux weakening characteristics
        plt.figure(figsize=(12, 8))
        self.load_field.plot_flux_weakening_characteristics()
        plt.savefig(f"{output_dir}/07_flux_weakening.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Turn interactive mode back on
        plt.ion()
    
    def design_optimization(self, objective: str = 'efficiency',
                          parameter_ranges: Dict = None) -> Dict:
        """
        Perform basic design optimization.
        
        Args:
            objective: Optimization objective ('efficiency', 'torque_density', 'power_density')
            parameter_ranges: Dictionary of parameter ranges to optimize
            
        Returns:
            Optimization results
        """
        print(f"\\nDESIGN OPTIMIZATION - Objective: {objective}")
        print("-" * 50)
        
        if parameter_ranges is None:
            # Default parameter ranges
            parameter_ranges = {
                'magnet_thickness': np.linspace(0.003, 0.008, 5),  # 3-8 mm
                'magnet_arc_ratio': np.linspace(0.6, 0.9, 5),     # 60-90%
                'current_angle': np.linspace(0, 45, 5)            # 0-45 degrees
            }
        
        best_value = 0
        best_params = {}
        results = []
        
        # Simple grid search
        for mag_thick in parameter_ranges.get('magnet_thickness', [self.magnet.magnet_thickness]):
            for arc_ratio in parameter_ranges.get('magnet_arc_ratio', [self.magnet.magnet_arc_ratio]):
                for curr_angle in parameter_ranges.get('current_angle', [30.0]):
                    
                    # Create modified geometry
                    temp_magnet = MagnetProperties(
                        residual_flux_density=self.magnet.residual_flux_density,
                        relative_permeability=self.magnet.relative_permeability,
                        coercivity=self.magnet.coercivity,
                        magnet_thickness=mag_thick,
                        magnet_arc_ratio=arc_ratio
                    )
                    
                    temp_load = LoadConditions(
                        rms_current=50.0,
                        current_angle=curr_angle,
                        frequency=50.0,
                        speed=1500.0,
                        load_torque=20.0
                    )
                    
                    # Create temporary analyzer
                    temp_analyzer = SlotlessMotorAnalyzer(
                        self.geometry, temp_magnet, self.winding, self.slot_geometry
                    )
                    
                    # Calculate performance
                    performance = temp_analyzer.load_field.motor_performance_analysis(temp_load)
                    torque = temp_analyzer.load_field.electromagnetic_torque(temp_load)
                    
                    # Calculate objective value
                    if objective == 'efficiency':
                        obj_value = performance.efficiency
                    elif objective == 'torque_density':
                        volume = np.pi * (self.geometry.stator_outer_radius**2 - 
                                        self.geometry.rotor_inner_radius**2) * self.geometry.axial_length
                        obj_value = torque / volume
                    elif objective == 'power_density':
                        volume = np.pi * (self.geometry.stator_outer_radius**2 - 
                                        self.geometry.rotor_inner_radius**2) * self.geometry.axial_length
                        obj_value = performance.output_power / volume
                    else:
                        obj_value = performance.efficiency
                    
                    results.append({
                        'magnet_thickness': mag_thick,
                        'magnet_arc_ratio': arc_ratio,
                        'current_angle': curr_angle,
                        'objective_value': obj_value,
                        'efficiency': performance.efficiency,
                        'torque': torque,
                        'power': performance.output_power
                    })
                    
                    if obj_value > best_value:
                        best_value = obj_value
                        best_params = {
                            'magnet_thickness': mag_thick,
                            'magnet_arc_ratio': arc_ratio,
                            'current_angle': curr_angle
                        }
        
        print(f"Best {objective}: {best_value:.4f}")
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value:.4f}")
        
        return {
            'best_value': best_value,
            'best_parameters': best_params,
            'all_results': results
        }
    
    def comparative_analysis(self, designs: List[Dict]) -> Dict:
        """
        Compare multiple motor designs.
        
        Args:
            designs: List of design dictionaries with motor parameters
            
        Returns:
            Comparison results
        """
        print("\\nCOMPARATIVE DESIGN ANALYSIS")
        print("-" * 50)
        
        comparison_results = []
        
        for i, design in enumerate(designs):
            print(f"\\nAnalyzing Design {i+1}...")
            
            # Extract parameters
            geometry = design.get('geometry', self.geometry)
            magnet = design.get('magnet', self.magnet)
            winding = design.get('winding', self.winding)
            load_cond = design.get('load_conditions', LoadConditions(50.0, 30.0, 50.0, 1500.0, 20.0))
            
            # Create analyzer for this design
            analyzer = SlotlessMotorAnalyzer(geometry, magnet, winding, self.slot_geometry)
            
            # Perform analysis
            results = analyzer.complete_analysis(load_cond, save_results=False)
            
            comparison_results.append({
                'design_id': i+1,
                'geometry': geometry,
                'magnet': magnet,
                'results': results
            })
        
        # Print comparison summary
        print("\\nCOMPARISON SUMMARY")
        print("-" * 30)
        print("Design | Efficiency | Torque | Power | THD")
        print("-" * 40)
        
        for result in comparison_results:
            design_id = result['design_id']
            efficiency = result['results']['load_analysis']['performance'].efficiency * 100
            torque = result['results']['load_analysis']['electromagnetic_torque']
            power = result['results']['load_analysis']['performance'].output_power
            thd = result['results']['open_circuit']['thd']
            
            print(f"{design_id:6d} | {efficiency:8.1f}% | {torque:6.1f} | {power:5.0f} | {thd:5.1f}%")
        
        return comparison_results


def main():
    """Main function demonstrating complete analysis workflow."""
    
    print("SLOTLESS MOTOR ELECTROMAGNETIC ANALYSIS")
    print("Based on Zhu's Mathematical Model")
    print("="*60)
    
    # Define a typical slotless PM motor
    geometry = MotorGeometry(
        stator_inner_radius=0.05,    # 50 mm
        stator_outer_radius=0.08,    # 80 mm
        rotor_outer_radius=0.048,    # 48 mm
        rotor_inner_radius=0.02,     # 20 mm
        airgap_length=0.002,         # 2 mm
        axial_length=0.1,            # 100 mm
        pole_pairs=4                 # 4 pole pairs (8 poles)
    )
    
    # High-energy permanent magnets (NdFeB)
    magnet = MagnetProperties(
        residual_flux_density=1.2,   # 1.2 T
        relative_permeability=1.05,
        coercivity=900000,           # 900 kA/m
        magnet_thickness=0.005,      # 5 mm
        magnet_arc_ratio=0.8         # 80% of pole pitch
    )
    
    # Three-phase distributed winding
    winding = WindingConfiguration(
        phases=3,
        slots_per_pole_per_phase=1.5,
        turns_per_coil=50,
        coil_span=0.8,               # 80% of pole pitch
        winding_factor=0.96,         # Typical for distributed winding
        current_density=5e6,         # 5 A/mm²
        conductor_area=2e-6          # 2 mm²
    )
    
    # Operating conditions
    load_conditions = LoadConditions(
        rms_current=50.0,            # 50 A RMS
        current_angle=30.0,          # 30° current advance angle
        frequency=50.0,              # 50 Hz
        speed=1500.0,                # 1500 rpm
        load_torque=20.0             # 20 Nm
    )
    
    # Create complete analyzer
    analyzer = SlotlessMotorAnalyzer(geometry, magnet, winding)
    
    # Perform complete analysis
    start_time = time.time()
    results = analyzer.complete_analysis(load_conditions, save_results=True)
    analysis_time = time.time() - start_time
    
    print(f"\\nTotal analysis time: {analysis_time:.2f} seconds")
    
    # Demonstrate design optimization
    print("\\n" + "="*60)
    optimization_results = analyzer.design_optimization(
        objective='efficiency',
        parameter_ranges={
            'magnet_thickness': np.linspace(0.004, 0.007, 3),
            'magnet_arc_ratio': np.linspace(0.7, 0.9, 3),
            'current_angle': np.linspace(20, 40, 3)
        }
    )
    
    print("\\nAnalysis complete! Check the 'results' directory for detailed plots.")


if __name__ == "__main__":
    main()