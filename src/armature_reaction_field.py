"""
Armature-Reaction Field Calculations (Zhu Part 2)

This module implements the mathematical model for armature reaction magnetic field
calculations in slotless permanent magnet motors based on Zhu's analytical methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
# Support both absolute and relative imports for notebook vs package usage
try:
    from open_circuit_field import MotorGeometry, WindingConfiguration
except ImportError: 
    from .open_circuit_field import MotorGeometry, WindingConfiguration


class ArmatureReactionField:
    """
    Armature reaction magnetic field calculation for slotless PM motors.
    
    Based on Zhu's analytical method using current sheet approximation
    and magnetic vector potential.
    """
    
    def __init__(self, geometry: MotorGeometry, winding: WindingConfiguration):
        self.geometry = geometry
        self.winding = winding
        self.mu0 = 4 * np.pi * 1e-7  # Permeability of free space
        
        # Calculate derived parameters
        self._calculate_winding_parameters()
    
    def _calculate_winding_parameters(self):
        """Calculate derived winding parameters"""
        self.pole_pitch = np.pi / self.geometry.pole_pairs
        
        # Total number of conductors per phase
        self.conductors_per_phase = (self.winding.turns_per_coil * 
                                   self.winding.slots_per_pole_per_phase * 
                                   self.geometry.pole_pairs * 2)
        
        # Linear current density amplitude
        self.current_amplitude = (self.winding.current_density * 
                                self.winding.conductor_area * 
                                self.conductors_per_phase)
    
    def current_sheet_distribution(self, theta: np.ndarray, time: float = 0, 
                                 current_magnitude: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Calculate the current sheet distribution for three-phase winding.
        
        Args:
            theta: Angular positions (rad)
            time: Time instant (s)
            current_magnitude: Current magnitude (A)
            
        Returns:
            Dictionary with current distributions for each phase
        """
        omega = 2 * np.pi * 50  # Assume 50 Hz for example
        p = self.geometry.pole_pairs
        
        # Three-phase currents
        ia = current_magnitude * np.cos(omega * time)
        ib = current_magnitude * np.cos(omega * time - 2*np.pi/3)
        ic = current_magnitude * np.cos(omega * time - 4*np.pi/3)
        
        # Current sheet distributions (simplified for demonstration)
        Ka = ia * self.winding.winding_factor * np.cos(p * theta)
        Kb = ib * self.winding.winding_factor * np.cos(p * theta - 2*np.pi/3)
        Kc = ic * self.winding.winding_factor * np.cos(p * theta - 4*np.pi/3)
        
        # Total current sheet
        K_total = Ka + Kb + Kc
        
        return {
            'phase_a': Ka,
            'phase_b': Kb, 
            'phase_c': Kc,
            'total': K_total,
            'currents': {'ia': ia, 'ib': ib, 'ic': ic}
        }
    
    def armature_reaction_coefficients(self, n_harmonics: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Fourier coefficients for armature reaction field.
        
        Args:
            n_harmonics: Number of harmonics to calculate
            
        Returns:
            Tuple of (harmonic_orders, coefficients)
        """
        # Only harmonics that are multiples of pole pairs
        harmonics = []
        coefficients = []
        
        p = self.geometry.pole_pairs
        
        for n in range(1, n_harmonics + 1):
            if n % p == 0:  # Only pole pair harmonics contribute
                # Winding factor for n-th harmonic
                kw_n = self.winding.winding_factor * np.sin(n * np.pi / (2 * p))
                
                # Coefficient for n-th harmonic
                coeff = (4 * self.mu0 * kw_n * self.current_amplitude) / (n * np.pi * p)
                
                harmonics.append(n)
                coefficients.append(coeff)
        
        return np.array(harmonics), np.array(coefficients)
    
    def radial_flux_density_armature(self, radius: np.ndarray, theta: np.ndarray,
                                   current_magnitude: float = 1.0, 
                                   time: float = 0,
                                   n_harmonics: int = 20) -> np.ndarray:
        """
        Calculate radial flux density due to armature reaction.
        
        Args:
            radius: Radial positions (m)
            theta: Angular positions (rad)
            current_magnitude: Current magnitude (A)
            time: Time instant (s)
            n_harmonics: Number of harmonics to include
            
        Returns:
            Radial flux density Br_arm(r,θ,t) (T)
        """
        harmonics, coeffs = self.armature_reaction_coefficients(n_harmonics)
        
        R, THETA = np.meshgrid(radius, theta, indexing='ij')
        Br_arm = np.zeros_like(R)
        
        Rs = self.geometry.stator_inner_radius
        Rr = self.geometry.rotor_outer_radius
        p = self.geometry.pole_pairs
        omega = 2 * np.pi * 50  # Fundamental frequency
        
        # Time-varying component
        current_factor = current_magnitude * np.cos(omega * time)
        
        for n, coeff in zip(harmonics, coeffs):
            if coeff != 0:
                # Radial dependence in airgap (similar to PM field but with different source)
                mask = (R >= Rr) & (R <= Rs)
                r_factor = np.ones_like(R)
                r_factor[mask] = ((R[mask]/Rs)**(n-1) + 
                                (Rr/Rs)**(2*n) * (R[mask]/Rs)**(-n-1)) / \
                               (1 + (Rr/Rs)**(2*n))
                
                Br_arm += coeff * current_factor * r_factor * np.cos(n * THETA)
        
        return Br_arm
    
    def tangential_flux_density_armature(self, radius: np.ndarray, theta: np.ndarray,
                                       current_magnitude: float = 1.0,
                                       time: float = 0,
                                       n_harmonics: int = 20) -> np.ndarray:
        """
        Calculate tangential flux density due to armature reaction.
        
        Args:
            radius: Radial positions (m)
            theta: Angular positions (rad)
            current_magnitude: Current magnitude (A)
            time: Time instant (s)
            n_harmonics: Number of harmonics to include
            
        Returns:
            Tangential flux density Bt_arm(r,θ,t) (T)
        """
        harmonics, coeffs = self.armature_reaction_coefficients(n_harmonics)
        
        R, THETA = np.meshgrid(radius, theta, indexing='ij')
        Bt_arm = np.zeros_like(R)
        
        Rs = self.geometry.stator_inner_radius
        Rr = self.geometry.rotor_outer_radius
        p = self.geometry.pole_pairs
        omega = 2 * np.pi * 50
        
        # Time-varying component
        current_factor = current_magnitude * np.cos(omega * time)
        
        for n, coeff in zip(harmonics, coeffs):
            if coeff != 0:
                # Radial dependence in airgap
                mask = (R >= Rr) & (R <= Rs)
                r_factor = np.ones_like(R)
                r_factor[mask] = ((R[mask]/Rs)**(n-1) - 
                                (Rr/Rs)**(2*n) * (R[mask]/Rs)**(-n-1)) / \
                               (1 + (Rr/Rs)**(2*n))
                
                Bt_arm += coeff * current_factor * r_factor * np.sin(n * THETA)
        
        return Bt_arm
    
    def armature_reaction_inductance(self, n_harmonics: int = 20) -> Dict[str, float]:
        """
        Calculate armature reaction inductances.
        
        Args:
            n_harmonics: Number of harmonics to include
            
        Returns:
            Dictionary with self and mutual inductances
        """
        harmonics, coeffs = self.armature_reaction_coefficients(n_harmonics)
        
        Rs = self.geometry.stator_inner_radius
        Rr = self.geometry.rotor_outer_radius
        L = self.geometry.axial_length
        p = self.geometry.pole_pairs
        
        # Self inductance calculation
        L_self = 0
        for n, coeff in zip(harmonics, coeffs):
            if n == p:  # Fundamental component
                # Permeance factor for airgap
                lambda_n = (1 - (Rr/Rs)**(2*n)) / (1 + (Rr/Rs)**(2*n))
                L_self += (self.mu0 * L * (self.winding.winding_factor)**2 * 
                          self.conductors_per_phase**2 * lambda_n) / (n * p)
        
        # Mutual inductance (for 3-phase symmetric winding)
        L_mutual = -L_self / 2
        
        return {
            'self_inductance': L_self,
            'mutual_inductance': L_mutual,
            'cyclic_inductance': L_self - L_mutual
        }
    
    def plot_current_sheet_distribution(self, time: float = 0, 
                                      current_magnitude: float = 100.0,
                                      figsize: Tuple[int, int] = (12, 8)):
        """
        Plot the current sheet distribution.
        
        Args:
            time: Time instant (s)
            current_magnitude: Current magnitude (A)
            figsize: Figure size
        """
        theta_deg = np.linspace(0, 360, 720)
        theta_rad = np.deg2rad(theta_deg)
        
        current_dist = self.current_sheet_distribution(theta_rad, time, current_magnitude)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Individual phase currents
        ax1.plot(theta_deg, current_dist['phase_a'], 'r-', label='Phase A', linewidth=2)
        ax1.plot(theta_deg, current_dist['phase_b'], 'g-', label='Phase B', linewidth=2)
        ax1.plot(theta_deg, current_dist['phase_c'], 'b-', label='Phase C', linewidth=2)
        ax1.set_xlabel('Angular position (degrees)')
        ax1.set_ylabel('Current sheet density (A/m)')
        ax1.set_title('Individual Phase Current Sheet Distributions')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Total current sheet
        ax2.plot(theta_deg, current_dist['total'], 'k-', linewidth=2, label='Total')
        ax2.set_xlabel('Angular position (degrees)')
        ax2.set_ylabel('Total current sheet density (A/m)')
        ax2.set_title('Total Current Sheet Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Print current values
        currents = current_dist['currents']
        print(f"Instantaneous currents at t = {time:.3f} s:")
        print(f"Phase A: {currents['ia']:.2f} A")
        print(f"Phase B: {currents['ib']:.2f} A") 
        print(f"Phase C: {currents['ic']:.2f} A")
        
        plt.tight_layout()
        plt.show()
    
    def plot_armature_flux_density(self, radius: Optional[float] = None,
                                 current_magnitude: float = 100.0,
                                 time: float = 0,
                                 n_harmonics: int = 20,
                                 figsize: Tuple[int, int] = (12, 8)):
        """
        Plot the armature reaction flux density distribution.
        
        Args:
            radius: Radius at which to plot (default: middle of airgap)
            current_magnitude: Current magnitude (A)
            time: Time instant (s)
            n_harmonics: Number of harmonics to include
            figsize: Figure size
        """
        if radius is None:
            radius = (self.geometry.rotor_outer_radius + self.geometry.stator_inner_radius) / 2
        
        theta_deg = np.linspace(0, 360, 720)
        theta_rad = np.deg2rad(theta_deg)
        
        Br = self.radial_flux_density_armature(np.array([radius]), theta_rad, 
                                             current_magnitude, time, n_harmonics)
        Bt = self.tangential_flux_density_armature(np.array([radius]), theta_rad,
                                                  current_magnitude, time, n_harmonics)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Radial flux density
        ax1.plot(theta_deg, Br[0, :], 'b-', linewidth=2, label='Radial flux density')
        ax1.set_xlabel('Angular position (degrees)')
        ax1.set_ylabel('Radial flux density (T)')
        ax1.set_title(f'Armature Reaction Radial Flux Density at r = {radius*1000:.1f} mm')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Tangential flux density
        ax2.plot(theta_deg, Bt[0, :], 'r-', linewidth=2, label='Tangential flux density')
        ax2.set_xlabel('Angular position (degrees)')
        ax2.set_ylabel('Tangential flux density (T)')
        ax2.set_title(f'Armature Reaction Tangential Flux Density at r = {radius*1000:.1f} mm')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def calculate_torque(self, current_magnitude: float, 
                        pm_flux_linkage: float) -> float:
        """
        Calculate electromagnetic torque due to armature reaction.
        
        Args:
            current_magnitude: RMS current magnitude (A)
            pm_flux_linkage: PM flux linkage (Wb)
            
        Returns:
            Electromagnetic torque (Nm)
        """
        # Simplified torque calculation
        # T = (3/2) * p * ψ_pm * I * cos(γ)
        # Assuming γ = 0 (current in phase with PM flux)
        
        torque = (3/2) * self.geometry.pole_pairs * pm_flux_linkage * current_magnitude
        
        return torque


def example_usage():
    """Example usage of the ArmatureReactionField class"""
    
    # Define motor geometry (same as in open circuit example)
    geometry = MotorGeometry(
        stator_inner_radius=0.05,    # 50 mm
        stator_outer_radius=0.08,    # 80 mm
        rotor_outer_radius=0.048,    # 48 mm
        rotor_inner_radius=0.02,     # 20 mm
        airgap_length=0.002,         # 2 mm
        axial_length=0.1,            # 100 mm
        pole_pairs=4
    )
    
    # Define winding configuration (rectangular wire)
    winding = WindingConfiguration(
        phases=3,
        slots_per_pole_per_phase=1.5,
        turns_per_coil=50,
        coil_span=0.8,              # 80% of pole pitch
        current_density=5e6,        # 5 A/mm²
        wire_width=1e-3,
        wire_height=2e-3,
    )
    
    # Create armature reaction field calculator
    ar_field = ArmatureReactionField(geometry, winding)
    
    # Plot current sheet distribution
    ar_field.plot_current_sheet_distribution(time=0, current_magnitude=100)
    
    # Plot armature reaction flux density
    ar_field.plot_armature_flux_density(current_magnitude=100, time=0)
    
    # Calculate inductances
    inductances = ar_field.armature_reaction_inductance()
    print("\nArmature Reaction Inductances:")
    print(f"Self inductance: {inductances['self_inductance']*1000:.2f} mH")
    print(f"Mutual inductance: {inductances['mutual_inductance']*1000:.2f} mH")
    print(f"Cyclic inductance: {inductances['cyclic_inductance']*1000:.2f} mH")


if __name__ == "__main__":
    example_usage()