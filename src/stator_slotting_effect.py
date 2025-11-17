"""
Stator Slotting Effect Calculations (Zhu Part 3)

This module implements the mathematical model for the effect of stator slotting
on magnetic field distribution in permanent magnet motors based on Zhu's analytical methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from .open_circuit_field import MotorGeometry


@dataclass
class SlotGeometry:
    """Stator slot geometry parameters"""
    slot_number: int              # Qs (total number of slots)
    slot_opening: float           # bs (slot opening width in m)
    slot_depth: float             # hs (slot depth in m)
    slot_width: float             # bw (slot width in m)
    tooth_width: float            # bt (tooth width in m)
    slot_shape: str               # 'rectangular', 'trapezoidal', 'round'


@dataclass
class SlottingParameters:
    """Parameters for slotting effect calculation"""
    permeance_method: str         # 'conformal_mapping', 'relative_permeance'
    harmonic_orders: List[int]    # Slot harmonic orders to consider
    slot_fill_factor: float       # Slot fill factor (0-1)


class StatorSlottingEffect:
    """
    Stator slotting effect calculation for PM motors.
    
    Based on Zhu's analytical method using relative permeance variation
    and conformal mapping techniques.
    """
    
    def __init__(self, geometry: MotorGeometry, slot_geometry: SlotGeometry):
        self.geometry = geometry
        self.slot_geometry = slot_geometry
        self.mu0 = 4 * np.pi * 1e-7  # Permeability of free space
        
        # Calculate derived parameters
        self._calculate_slotting_parameters()
    
    def _calculate_slotting_parameters(self):
        """Calculate derived slotting parameters"""
        self.slot_pitch = 2 * np.pi / self.slot_geometry.slot_number
        self.tooth_pitch = self.slot_pitch
        
        # Slot opening coefficient
        self.slot_opening_coeff = (self.slot_geometry.slot_opening / 
                                  (self.geometry.stator_inner_radius * self.slot_pitch))
        
        # Carter's coefficient
        self.carter_coefficient = self._calculate_carter_coefficient()
        
        # Effective airgap
        self.effective_airgap = self.geometry.airgap_length * self.carter_coefficient
    
    def _calculate_carter_coefficient(self) -> float:
        """
        Calculate Carter's coefficient for airgap extension due to slotting.
        
        Returns:
            Carter's coefficient kc
        """
        # Simplified Carter's coefficient calculation
        bs = self.slot_geometry.slot_opening
        g = self.geometry.airgap_length
        tau_s = self.geometry.stator_inner_radius * self.slot_pitch
        
        # Carter's coefficient for rectangular slots
        if self.slot_geometry.slot_shape == 'rectangular':
            gamma = (4/np.pi) * np.arctan(bs / (2*g)) - bs / tau_s
            kc = tau_s / (tau_s - gamma * bs)
        else:
            # Simplified approximation for other slot shapes
            kc = 1 + (bs / g) * (1 / (5 + bs/g))
        
        return kc
    
    def relative_permeance_function(self, theta: np.ndarray) -> np.ndarray:
        """
        Calculate the relative permeance function due to slotting.
        
        Args:
            theta: Angular positions (rad)
            
        Returns:
            Relative permeance λ(θ)
        """
        Qs = self.slot_geometry.slot_number
        bs = self.slot_geometry.slot_opening
        Rs = self.geometry.stator_inner_radius
        
        # Fundamental permeance
        lambda_0 = 1.0
        
        # Slotting permeance harmonics
        lambda_slot = np.zeros_like(theta)
        
        # Include first few significant slot harmonics
        for n in range(1, 10):  # First 10 slot harmonics
            # Slot harmonic amplitude
            if n * Qs != 0:
                An = (2 * bs / (Rs * self.slot_pitch * n * np.pi)) * \
                     np.sin(n * np.pi * bs / (Rs * self.slot_pitch))
                
                lambda_slot += An * np.cos(n * Qs * theta)
        
        # Total relative permeance
        lambda_total = lambda_0 + lambda_slot
        
        return lambda_total
    
    def slotting_effect_on_radial_field(self, radius: np.ndarray, theta: np.ndarray,
                                      br_no_slots: np.ndarray) -> np.ndarray:
        """
        Calculate the effect of slotting on radial magnetic field.
        
        Args:
            radius: Radial positions (m)
            theta: Angular positions (rad)
            br_no_slots: Radial flux density without slots (T)
            
        Returns:
            Modified radial flux density with slotting effect (T)
        """
        # Relative permeance function
        lambda_rel = self.relative_permeance_function(theta)
        
        # Apply slotting effect
        R, THETA = np.meshgrid(radius, theta, indexing='ij')
        LAMBDA = np.broadcast_to(lambda_rel, R.shape)
        
        # Slotting effect diminishes with distance from stator surface
        Rs = self.geometry.stator_inner_radius
        distance_factor = np.exp(-(Rs - R) / self.geometry.airgap_length)
        
        # Modified field
        br_with_slots = br_no_slots * (1 + (LAMBDA - 1) * distance_factor)
        
        return br_with_slots
    
    def slot_harmonic_analysis(self, n_harmonics: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze slot harmonics in the permeance function.
        
        Args:
            n_harmonics: Number of harmonics to analyze
            
        Returns:
            Tuple of (harmonic_orders, harmonic_amplitudes)
        """
        theta = np.linspace(0, 2*np.pi, 2048)
        lambda_rel = self.relative_permeance_function(theta)
        
        # Remove DC component
        lambda_ac = lambda_rel - np.mean(lambda_rel)
        
        # FFT analysis
        fft_result = np.fft.fft(lambda_ac)
        harmonic_amplitudes = 2 * np.abs(fft_result[:n_harmonics]) / len(theta)
        
        # Slot harmonic orders
        Qs = self.slot_geometry.slot_number
        slot_harmonics = []
        slot_amplitudes = []
        
        for n in range(1, n_harmonics):
            if n % Qs == 0 or abs(n % Qs - Qs) < 3:  # Slot-related harmonics
                slot_harmonics.append(n)
                slot_amplitudes.append(harmonic_amplitudes[n])
        
        return np.array(slot_harmonics), np.array(slot_amplitudes)
    
    def cogging_torque_calculation(self, pm_flux_density: np.ndarray, 
                                 theta_rotor: np.ndarray) -> np.ndarray:
        """
        Calculate cogging torque due to slotting effect.
        
        Args:
            pm_flux_density: PM flux density distribution
            theta_rotor: Rotor angular positions (rad)
            
        Returns:
            Cogging torque as function of rotor position (Nm)
        """
        Qs = self.slot_geometry.slot_number
        p = self.geometry.pole_pairs
        Rs = self.geometry.stator_inner_radius
        L = self.geometry.axial_length
        
        # Cogging torque harmonics
        cogging_torque = np.zeros_like(theta_rotor)
        
        # Calculate energy method for cogging torque
        for n in range(1, 10):  # First few cogging harmonics
            # Cogging harmonic order
            nc = n * Qs
            
            if nc != 0:
                # Cogging amplitude (simplified)
                Tc_n = (np.pi * Rs**2 * L / (4 * self.mu0)) * \
                       (pm_flux_density.max())**2 * \
                       self.slot_opening_coeff * \
                       np.sin(nc * np.pi / (2 * p))
                
                # Cogging torque harmonic
                cogging_torque += Tc_n * np.sin(nc * theta_rotor / p)
        
        return cogging_torque
    
    def plot_permeance_function(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot the relative permeance function.
        
        Args:
            figsize: Figure size
        """
        theta_deg = np.linspace(0, 360, 1440)
        theta_rad = np.deg2rad(theta_deg)
        
        lambda_rel = self.relative_permeance_function(theta_rad)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        ax.plot(theta_deg, lambda_rel, 'b-', linewidth=2)
        ax.set_xlabel('Angular position (degrees)')
        ax.set_ylabel('Relative permeance λ(θ)')
        ax.set_title('Relative Permeance Function due to Stator Slotting')
        ax.grid(True, alpha=0.3)
        
        # Mark slot positions
        slot_positions = np.arange(0, 360, 360/self.slot_geometry.slot_number)
        for pos in slot_positions:
            ax.axvline(x=pos, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_slotting_effect_comparison(self, br_no_slots: np.ndarray,
                                      radius: Optional[float] = None,
                                      figsize: Tuple[int, int] = (12, 8)):
        """
        Plot comparison of flux density with and without slotting effect.
        
        Args:
            br_no_slots: Radial flux density without slots
            radius: Radius at which to plot (default: stator inner radius)
            figsize: Figure size
        """
        if radius is None:
            radius = self.geometry.stator_inner_radius
        
        theta_deg = np.linspace(0, 360, 720)
        theta_rad = np.deg2rad(theta_deg)
        
        # Apply slotting effect
        br_with_slots = self.slotting_effect_on_radial_field(
            np.array([radius]), theta_rad, br_no_slots
        )
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Comparison plot
        ax1.plot(theta_deg, br_no_slots[0, :], 'b-', linewidth=2, 
                label='Without slotting')
        ax1.plot(theta_deg, br_with_slots[0, :], 'r-', linewidth=2, 
                label='With slotting')
        ax1.set_xlabel('Angular position (degrees)')
        ax1.set_ylabel('Radial flux density (T)')
        ax1.set_title(f'Slotting Effect on Radial Flux Density at r = {radius*1000:.1f} mm')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Difference plot
        difference = br_with_slots[0, :] - br_no_slots[0, :]
        ax2.plot(theta_deg, difference, 'g-', linewidth=2)
        ax2.set_xlabel('Angular position (degrees)')
        ax2.set_ylabel('Flux density difference (T)')
        ax2.set_title('Difference due to Slotting Effect')
        ax2.grid(True, alpha=0.3)
        
        # Mark slot positions
        slot_positions = np.arange(0, 360, 360/self.slot_geometry.slot_number)
        for pos in slot_positions:
            ax1.axvline(x=pos, color='k', linestyle='--', alpha=0.3)
            ax2.axvline(x=pos, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cogging_torque(self, pm_flux_density_max: float = 1.0,
                           figsize: Tuple[int, int] = (12, 6)):
        """
        Plot cogging torque variation.
        
        Args:
            pm_flux_density_max: Maximum PM flux density (T)
            figsize: Figure size
        """
        # Rotor positions for one cogging period
        cogging_period = 2 * np.pi / (self.slot_geometry.slot_number * self.geometry.pole_pairs)
        theta_rotor = np.linspace(0, cogging_period, 1000)
        
        # Simplified PM flux density
        pm_flux = np.ones((1, 1000)) * pm_flux_density_max
        
        # Calculate cogging torque
        cogging_torque = self.cogging_torque_calculation(pm_flux, theta_rotor)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        ax.plot(np.rad2deg(theta_rotor), cogging_torque, 'r-', linewidth=2)
        ax.set_xlabel('Rotor position (degrees)')
        ax.set_ylabel('Cogging torque (Nm)')
        ax.set_title('Cogging Torque due to Stator Slotting')
        ax.grid(True, alpha=0.3)
        
        # Statistics
        print(f"Peak-to-peak cogging torque: {np.ptp(cogging_torque):.4f} Nm")
        print(f"RMS cogging torque: {np.sqrt(np.mean(cogging_torque**2)):.4f} Nm")
        
        plt.tight_layout()
        plt.show()
    
    def calculate_slot_leakage_inductance(self) -> float:
        """
        Calculate slot leakage inductance.
        
        Returns:
            Slot leakage inductance (H)
        """
        Qs = self.slot_geometry.slot_number
        L = self.geometry.axial_length
        hs = self.slot_geometry.slot_depth
        bs = self.slot_geometry.slot_opening
        bw = self.slot_geometry.slot_width
        
        # Slot permeance (simplified for rectangular slot)
        lambda_slot = (hs / (3 * bw)) + (self.geometry.airgap_length / bs)
        
        # Turns per slot (assuming distributed winding)
        turns_per_slot = 50  # Example value
        
        # Slot leakage inductance
        L_slot = self.mu0 * turns_per_slot**2 * lambda_slot * L
        
        return L_slot


def example_usage():
    """Example usage of the StatorSlottingEffect class"""
    
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
    
    # Define slot geometry
    slot_geometry = SlotGeometry(
        slot_number=24,              # 24 slots
        slot_opening=0.003,          # 3 mm
        slot_depth=0.015,            # 15 mm
        slot_width=0.005,            # 5 mm
        tooth_width=0.008,           # 8 mm
        slot_shape='rectangular'
    )
    
    # Create slotting effect calculator
    slotting = StatorSlottingEffect(geometry, slot_geometry)
    
    # Plot permeance function
    slotting.plot_permeance_function()
    
    # Create example flux density without slots (sinusoidal)
    theta = np.linspace(0, 2*np.pi, 720)
    br_no_slots = np.array([np.sin(4 * theta)])  # 4 pole pairs
    
    # Plot slotting effect comparison
    slotting.plot_slotting_effect_comparison(br_no_slots)
    
    # Plot cogging torque
    slotting.plot_cogging_torque(pm_flux_density_max=1.2)
    
    # Calculate slot leakage inductance
    L_slot = slotting.calculate_slot_leakage_inductance()
    print(f"\nSlot leakage inductance: {L_slot*1000:.2f} mH")
    
    # Analyze slot harmonics
    harmonics, amplitudes = slotting.slot_harmonic_analysis()
    print("\nSlot Harmonic Analysis:")
    print("Harmonic Order | Amplitude")
    print("-" * 25)
    for h, amp in zip(harmonics[:5], amplitudes[:5]):
        print(f"{h:12d} | {amp:10.4f}")


if __name__ == "__main__":
    example_usage()