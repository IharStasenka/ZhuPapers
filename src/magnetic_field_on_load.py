"""
Magnetic Field on Load Calculations (Zhu Part 4)

This module implements the mathematical model for magnetic field distribution
under load conditions in permanent magnet motors based on Zhu's analytical methods.
This combines the effects of permanent magnets, armature reaction, and slotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from .open_circuit_field import MotorGeometry, MagnetProperties, OpenCircuitField
from .armature_reaction_field import WindingConfiguration, ArmatureReactionField
from .stator_slotting_effect import SlotGeometry, StatorSlottingEffect


@dataclass
class LoadConditions:
    """Operating load conditions"""
    rms_current: float            # RMS phase current (A)
    current_angle: float          # Current angle relative to PM flux (degrees)
    frequency: float              # Electrical frequency (Hz)
    speed: float                  # Mechanical speed (rpm)
    load_torque: float            # Load torque (Nm)


@dataclass
class MotorPerformance:
    """Motor performance parameters"""
    efficiency: float
    power_factor: float
    input_power: float            # Input power (W)
    output_power: float           # Output power (W)
    copper_losses: float          # Copper losses (W)
    iron_losses: float            # Iron losses (W)


class MagneticFieldOnLoad:
    """
    Magnetic field calculation under load conditions for PM motors.
    
    Combines permanent magnet field, armature reaction field, and slotting effects
    based on Zhu's analytical methods.
    """
    
    def __init__(self, geometry: MotorGeometry, magnet: MagnetProperties,
                 winding: WindingConfiguration, slot_geometry: Optional[SlotGeometry] = None):
        self.geometry = geometry
        self.magnet = magnet
        self.winding = winding
        self.slot_geometry = slot_geometry
        
        # Initialize component calculators
        self.oc_field = OpenCircuitField(geometry, magnet)
        self.ar_field = ArmatureReactionField(geometry, winding)
        if slot_geometry:
            self.slotting = StatorSlottingEffect(geometry, slot_geometry)
        else:
            self.slotting = None
        
        self.mu0 = 4 * np.pi * 1e-7  # Permeability of free space
    
    def total_radial_flux_density(self, radius: np.ndarray, theta: np.ndarray,
                                 load_conditions: LoadConditions,
                                 time: float = 0,
                                 include_slotting: bool = True,
                                 n_harmonics: int = 20) -> np.ndarray:
        """
        Calculate total radial flux density under load conditions.
        
        Args:
            radius: Radial positions (m)
            theta: Angular positions (rad)
            load_conditions: Load operating conditions
            time: Time instant (s)
            include_slotting: Whether to include slotting effects
            n_harmonics: Number of harmonics to include
            
        Returns:
            Total radial flux density Br_total(r,θ,t) (T)
        """
        # PM open-circuit field
        Br_pm = self.oc_field.radial_flux_density(radius, theta, n_harmonics)
        
        # Armature reaction field
        omega = 2 * np.pi * load_conditions.frequency
        current_phase = np.deg2rad(load_conditions.current_angle)
        current_time_factor = np.cos(omega * time + current_phase)
        
        Br_arm = self.ar_field.radial_flux_density_armature(
            radius, theta, load_conditions.rms_current, time, n_harmonics
        )
        
        # Combine PM and armature reaction
        Br_total = Br_pm + Br_arm
        
        # Apply slotting effect if requested and available
        if include_slotting and self.slotting:
            Br_total = self.slotting.slotting_effect_on_radial_field(
                radius, theta, Br_total
            )
        
        return Br_total
    
    def total_tangential_flux_density(self, radius: np.ndarray, theta: np.ndarray,
                                    load_conditions: LoadConditions,
                                    time: float = 0,
                                    include_slotting: bool = True,
                                    n_harmonics: int = 20) -> np.ndarray:
        """
        Calculate total tangential flux density under load conditions.
        
        Args:
            radius: Radial positions (m)
            theta: Angular positions (rad)
            load_conditions: Load operating conditions
            time: Time instant (s)
            include_slotting: Whether to include slotting effects
            n_harmonics: Number of harmonics to include
            
        Returns:
            Total tangential flux density Bt_total(r,θ,t) (T)
        """
        # PM open-circuit field
        Bt_pm = self.oc_field.tangential_flux_density(radius, theta, n_harmonics)
        
        # Armature reaction field
        Bt_arm = self.ar_field.tangential_flux_density_armature(
            radius, theta, load_conditions.rms_current, time, n_harmonics
        )
        
        # Combine PM and armature reaction
        Bt_total = Bt_pm + Bt_arm
        
        # Note: Slotting effect on tangential component is typically smaller
        # and not implemented in this simplified model
        
        return Bt_total
    
    def electromagnetic_torque(self, load_conditions: LoadConditions,
                             time: float = 0) -> float:
        """
        Calculate electromagnetic torque under load conditions.
        
        Args:
            load_conditions: Load operating conditions
            time: Time instant (s)
            
        Returns:
            Electromagnetic torque (Nm)
        """
        # PM flux linkage (simplified)
        # Include winding factor to reflect coil distribution/short pitch effects
        pm_flux_linkage = (
            self.oc_field.flux_linkage_per_turn(self.geometry.stator_inner_radius)
            * self.winding.turns_per_coil
            * self.winding.phases
            * getattr(self.winding, "winding_factor", 1.0)
        )
        
        # Current angle in radians
        gamma = np.deg2rad(load_conditions.current_angle)
        
        # Electromagnetic torque (considering current angle)
        Te = (
            1.5
            * self.geometry.pole_pairs
            * pm_flux_linkage
            * load_conditions.rms_current
            * np.cos(gamma)
        )
        
        return Te
    
    def flux_weakening_analysis(self, current_range: np.ndarray,
                              current_angle: float = 90.0) -> Dict[str, np.ndarray]:
        """
        Analyze flux weakening effect at different current levels.
        
        Args:
            current_range: Range of current values to analyze (A)
            current_angle: Current angle for flux weakening (degrees)
            
        Returns:
            Dictionary with flux weakening analysis results
        """
        results = {
            'current': current_range,
            'flux_linkage': np.zeros_like(current_range),
            'torque': np.zeros_like(current_range),
            'flux_density_peak': np.zeros_like(current_range)
        }
        
        # Reference radius for analysis
        r_ref = (self.geometry.rotor_outer_radius + self.geometry.stator_inner_radius) / 2
        theta_ref = np.linspace(0, 2*np.pi, 360)
        
        for i, current in enumerate(current_range):
            load_cond = LoadConditions(
                rms_current=current,
                current_angle=current_angle,
                frequency=50.0,
                speed=1500.0,
                load_torque=0.0
            )
            
            # Calculate total flux density
            Br_total = self.total_radial_flux_density(
                np.array([r_ref]), theta_ref, load_cond, include_slotting=False
            )
            
            # Peak flux density
            results['flux_density_peak'][i] = np.max(np.abs(Br_total))
            
            # Approximate flux linkage
            results['flux_linkage'][i] = np.trapz(Br_total[0, :], theta_ref) * \
                                       r_ref * self.geometry.axial_length
            
            # Torque
            results['torque'][i] = self.electromagnetic_torque(load_cond)
        
        return results
    
    def iron_loss_calculation(self, load_conditions: LoadConditions,
                            steinmetz_coefficients: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate iron losses using Steinmetz equation.
        
        Args:
            load_conditions: Load operating conditions
            steinmetz_coefficients: Steinmetz equation coefficients
            
        Returns:
            Dictionary with iron loss components
        """
        if steinmetz_coefficients is None:
            # Default coefficients for electrical steel
            steinmetz_coefficients = {
                'kh': 100.0,     # Hysteresis coefficient
                'kc': 0.5,       # Eddy current coefficient
                'alpha': 1.6,    # Frequency exponent
                'beta': 2.0      # Flux density exponent
            }
        
        # Calculate flux density at different radii
        radii = np.linspace(self.geometry.rotor_outer_radius, 
                           self.geometry.stator_inner_radius, 10)
        theta = np.linspace(0, 2*np.pi, 360)
        
        total_iron_loss = 0.0
        hysteresis_loss = 0.0
        eddy_current_loss = 0.0
        
        for r in radii:
            # Flux density at this radius
            Br = self.total_radial_flux_density(
                np.array([r]), theta, load_conditions, include_slotting=False
            )
            
            # RMS flux density
            B_rms = np.sqrt(np.mean(Br[0, :]**2))
            
            # Volume element
            dV = 2 * np.pi * r * (radii[1] - radii[0] if len(radii) > 1 else 0.001) * \
                 self.geometry.axial_length
            
            # Steinmetz equation
            f = load_conditions.frequency
            kh = steinmetz_coefficients['kh']
            kc = steinmetz_coefficients['kc']
            alpha = steinmetz_coefficients['alpha']
            beta = steinmetz_coefficients['beta']
            
            # Losses per unit volume
            ph = kh * (f**alpha) * (B_rms**beta)  # Hysteresis loss density
            pc = kc * (f**2) * (B_rms**2)         # Eddy current loss density
            
            # Total losses
            hysteresis_loss += ph * dV
            eddy_current_loss += pc * dV
        
        total_iron_loss = hysteresis_loss + eddy_current_loss
        
        return {
            'total_iron_loss': total_iron_loss,
            'hysteresis_loss': hysteresis_loss,
            'eddy_current_loss': eddy_current_loss
        }
    
    def motor_performance_analysis(self, load_conditions: LoadConditions) -> MotorPerformance:
        """
        Comprehensive motor performance analysis under load.
        
        Args:
            load_conditions: Load operating conditions
            
        Returns:
            Motor performance parameters
        """
        # Electromagnetic torque
        Te = self.electromagnetic_torque(load_conditions)
        
        # Mechanical power output
        omega_mech = 2 * np.pi * load_conditions.speed / 60  # rad/s
        P_out = Te * omega_mech
        
        # Copper losses (simplified)
        # Phase resistance (estimated)
        Rs = 0.5  # Ohm (example value)
        P_copper = 3 * (load_conditions.rms_current**2) * Rs
        
        # Iron losses
        iron_losses = self.iron_loss_calculation(load_conditions)
        P_iron = iron_losses['total_iron_loss']
        
        # Input power
        P_in = P_out + P_copper + P_iron
        
        # Efficiency
        efficiency = P_out / P_in if P_in > 0 else 0
        
        # Power factor (simplified calculation)
        # This would require more detailed analysis in practice
        power_factor = 0.85  # Typical value
        
        return MotorPerformance(
            efficiency=efficiency,
            power_factor=power_factor,
            input_power=P_in,
            output_power=P_out,
            copper_losses=P_copper,
            iron_losses=P_iron
        )
    
    def plot_load_field_distribution(self, load_conditions: LoadConditions,
                                   radius: Optional[float] = None,
                                   time: float = 0,
                                   include_slotting: bool = True,
                                   figsize: Tuple[int, int] = (15, 10)):
        """
        Plot magnetic field distribution under load conditions.
        
        Args:
            load_conditions: Load operating conditions
            radius: Radius at which to plot (default: middle of airgap)
            time: Time instant (s)
            include_slotting: Whether to include slotting effects
            figsize: Figure size
        """
        if radius is None:
            radius = (self.geometry.rotor_outer_radius + self.geometry.stator_inner_radius) / 2
        
        theta_deg = np.linspace(0, 360, 720)
        theta_rad = np.deg2rad(theta_deg)
        
        # Calculate individual components
        Br_pm = self.oc_field.radial_flux_density(np.array([radius]), theta_rad)
        Br_arm = self.ar_field.radial_flux_density_armature(
            np.array([radius]), theta_rad, load_conditions.rms_current, time
        )
        Br_total = self.total_radial_flux_density(
            np.array([radius]), theta_rad, load_conditions, time, include_slotting
        )
        
        Bt_total = self.total_tangential_flux_density(
            np.array([radius]), theta_rad, load_conditions, time, include_slotting
        )
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # PM field component
        ax1.plot(theta_deg, Br_pm[0, :], 'b-', linewidth=2, label='PM field')
        ax1.set_xlabel('Angular position (degrees)')
        ax1.set_ylabel('Radial flux density (T)')
        ax1.set_title('PM Open-Circuit Field')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Armature reaction component
        ax2.plot(theta_deg, Br_arm[0, :], 'r-', linewidth=2, label='Armature reaction')
        ax2.set_xlabel('Angular position (degrees)')
        ax2.set_ylabel('Radial flux density (T)')
        ax2.set_title('Armature Reaction Field')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Total radial field
        ax3.plot(theta_deg, Br_pm[0, :], 'b--', alpha=0.7, label='PM only')
        ax3.plot(theta_deg, Br_total[0, :], 'k-', linewidth=2, label='Total field')
        ax3.set_xlabel('Angular position (degrees)')
        ax3.set_ylabel('Radial flux density (T)')
        ax3.set_title('Total Radial Field on Load')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Total tangential field
        ax4.plot(theta_deg, Bt_total[0, :], 'g-', linewidth=2, label='Tangential field')
        ax4.set_xlabel('Angular position (degrees)')
        ax4.set_ylabel('Tangential flux density (T)')
        ax4.set_title('Tangential Field on Load')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print load information
        print(f"Load Conditions:")
        print(f"RMS Current: {load_conditions.rms_current:.2f} A")
        print(f"Current Angle: {load_conditions.current_angle:.1f}°")
        print(f"Frequency: {load_conditions.frequency:.1f} Hz")
        print(f"Speed: {load_conditions.speed:.0f} rpm")
    
    def plot_flux_weakening_characteristics(self, max_current: float = 200.0,
                                          figsize: Tuple[int, int] = (12, 8)):
        """
        Plot flux weakening characteristics.
        
        Args:
            max_current: Maximum current for analysis (A)
            figsize: Figure size
        """
        current_range = np.linspace(0, max_current, 50)
        
        # Analyze flux weakening (d-axis current, 90° phase angle)
        fw_results = self.flux_weakening_analysis(current_range, current_angle=90.0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Flux linkage vs current
        ax1.plot(current_range, np.abs(fw_results['flux_linkage']), 'b-', linewidth=2)
        ax1.set_xlabel('d-axis Current (A)')
        ax1.set_ylabel('Flux Linkage Magnitude (Wb)')
        ax1.set_title('Flux Weakening Characteristic')
        ax1.grid(True, alpha=0.3)
        
        # Peak flux density vs current
        ax2.plot(current_range, fw_results['flux_density_peak'], 'r-', linewidth=2)
        ax2.set_xlabel('d-axis Current (A)')
        ax2.set_ylabel('Peak Flux Density (T)')
        ax2.set_title('Peak Flux Density vs d-axis Current')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def example_usage():
    """Example usage of the MagneticFieldOnLoad class"""
    
    # Define motor geometry
    geometry = MotorGeometry(
        stator_inner_radius=0.05,    # 50 mm
        stator_outer_radius=0.08,    # 80 mm
        rotor_outer_radius=0.048,    # 48 mm
        rotor_inner_radius=0.02,     # 20 mm
        airgap_length=0.002,         # 2 mm
        axial_length=0.1,            # 100 mm
        pole_pairs=4
    )
    
    # Define magnet properties
    magnet = MagnetProperties(
        residual_flux_density=1.2,   # 1.2 T
        relative_permeability=1.05,
        coercivity=900000,           # 900 kA/m
        magnet_thickness=0.005,      # 5 mm
        magnet_arc_ratio=0.8         # 80% of pole pitch
    )
    
    # Define winding configuration (rectangular single-layer slotless wire)
    winding = WindingConfiguration(
        phases=3,
        slots_per_pole_per_phase=1.5,
        turns_per_coil=50,
        coil_span=0.8,
        current_density=5e6,         # 5 A/mm²
        wire_width=1e-3,
        wire_height=2e-3,
    )
    
    # Define load conditions
    load_conditions = LoadConditions(
        rms_current=50.0,            # 50 A RMS
        current_angle=30.0,          # 30° current angle
        frequency=50.0,              # 50 Hz
        speed=1500.0,                # 1500 rpm
        load_torque=20.0             # 20 Nm
    )
    
    # Create magnetic field on load calculator
    load_field = MagneticFieldOnLoad(geometry, magnet, winding)
    
    # Plot field distribution under load
    load_field.plot_load_field_distribution(load_conditions)
    
    # Calculate performance
    performance = load_field.motor_performance_analysis(load_conditions)
    print(f"\nMotor Performance Analysis:")
    print(f"Efficiency: {performance.efficiency*100:.1f}%")
    print(f"Input Power: {performance.input_power:.1f} W")
    print(f"Output Power: {performance.output_power:.1f} W")
    print(f"Copper Losses: {performance.copper_losses:.1f} W")
    print(f"Iron Losses: {performance.iron_losses:.1f} W")
    
    # Plot flux weakening characteristics
    load_field.plot_flux_weakening_characteristics()
    
    # Calculate electromagnetic torque
    torque = load_field.electromagnetic_torque(load_conditions)
    print(f"\nElectromagnetic Torque: {torque:.2f} Nm")


if __name__ == "__main__":
    example_usage()