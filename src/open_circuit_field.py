"""
Open-Circuit Field Calculations (Zhu Part 1)

This module implements the mathematical model for open-circuit magnetic field
calculations in slotless permanent magnet motors based on Zhu's analytical methods.

Key Equations Implemented:
- Equation (8): Magnetization distribution Mr(θ) = Σ Mn cos(nθ)
- Equation (9): Fourier coefficients Mn = (4Br)/(nπμr) * sin(nαm/2)
- Equations (15-16): Radial flux density Br(r,θ) with radial variation fn(r)
- Equations (17-18): Tangential flux density Bt(r,θ) with radial variation gn(r)
- Equation (19): Flux linkage calculation ψ = ∫ Br(r,θ) * r * L * dθ

Reference: Zhu, Z.Q., Howe, D. "Analytical prediction of the magnetic field
in the air-gap of surface-mounted permanent-magnet motors" - Part 1
"""

import numpy as np
# Alternate alias 'npy' for clarity when using 'n_p' variable for harmonic order
npy = np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class MotorGeometry:
    """Motor geometry parameters.

    stator_inner_radius: coil inner / airgap outer boundary.
    coil_thickness: slotless coil region (treated as μ≈μ0 for extended gap calculations).
    """
    stator_inner_radius: float
    stator_outer_radius: float
    rotor_outer_radius: float
    rotor_inner_radius: float
    airgap_length: float
    axial_length: float
    pole_pairs: int
    coil_thickness: float = 0.0

    @classmethod
    def from_thicknesses(
        cls,
        *,
        stator_outer_diameter: float,
        stator_thickness: float,
        coil_thickness: float = 0.0,
        airgap_length: float,
        magnet_thickness: float,
        rotor_thickness: float,
        axial_length: float,
        pole_pairs: int,
        min_shaft_radius: float = 0.0,
    ) -> "MotorGeometry":
        Rso = stator_outer_diameter / 2.0
        Rco_outer = Rso - stator_thickness
        if Rco_outer <= 0:
            raise ValueError("Invalid geometry: stator thickness exceeds outer radius.")

        Rs = Rco_outer - coil_thickness
        if Rs <= 0:
            raise ValueError("Invalid geometry: coil thickness + stator thickness exceed outer radius.")

        Rg_in = Rs - airgap_length
        if Rg_in <= 0 or Rg_in >= Rs:
            raise ValueError("Invalid geometry: airgap results in non-physical rotor/stator spacing.")

        Rm_in = Rg_in - magnet_thickness
        if Rm_in <= 0:
            raise ValueError("Invalid geometry: magnet thickness too large for available space.")

        Rri = Rm_in - rotor_thickness
        if Rri < min_shaft_radius:
            raise ValueError(
                f"Invalid geometry: rotor inner radius {Rri:.6f} < minimum shaft radius {min_shaft_radius:.6f}."
            )

        if not (Rri < Rm_in < Rg_in < Rs < Rso):
            raise ValueError("Invalid geometry: radii order violated (Rri < Rm_in < Rg_in < Rs < Rso).")

        return cls(
            stator_inner_radius=Rs,
            stator_outer_radius=Rso,
            rotor_outer_radius=Rg_in,
            rotor_inner_radius=Rri,
            airgap_length=airgap_length,
            axial_length=axial_length,
            pole_pairs=pole_pairs,
            coil_thickness=coil_thickness,
        )

    def layer_radii(
        self,
        magnet_thickness: float,
        stator_thickness: float = None,
        coil_thickness: float = 0.0,
    ) -> dict:
        Rs = self.stator_inner_radius
        Rso = self.stator_outer_radius
        Rg_in = self.rotor_outer_radius
        Rm_in = Rg_in - magnet_thickness
        if stator_thickness is not None:
            Rco_outer = Rso - stator_thickness
        else:
            Rco_outer = (Rso + Rs) / 2 if coil_thickness == 0 else Rs + coil_thickness
        return {
            "Rs": Rs,
            "Rg_in": Rg_in,
            "Rm_in": Rm_in,
            "Rri": self.rotor_inner_radius,
            "Rso": Rso,
            "Rco_outer": Rco_outer,
            "coil_thickness": coil_thickness,
        }


@dataclass
class MagnetProperties:
    """Permanent magnet properties"""
    residual_flux_density: float    # Br (T)
    relative_permeability: float    # μr
    coercivity: float               # Hc (A/m)
    magnet_thickness: float         # hm (m)
    magnet_arc_ratio: float         # αm (magnet arc to pole pitch ratio)


class OpenCircuitField:
    """
    Open-circuit magnetic field calculation for slotless PM motors.
    
    Based on Zhu's analytical method using magnetic scalar potential
    and Fourier series expansion.
    
    Mathematical Foundation:
    - Uses separation of variables in cylindrical coordinates
    - Magnetic scalar potential: Φ(r,θ) = Σ An * fn(r) * cos(nθ)
    - Boundary conditions at rotor and stator surfaces
    - Fourier series expansion of magnetization pattern
    
    Key Assumptions:
    - Linear magnetic materials (μr = constant)
    - Slotless stator (smooth airgap)
    - Radial magnetization of surface-mounted magnets
    - Infinite axial length (2D analysis)
    """
    
    def __init__(self, geometry: MotorGeometry, magnet: MagnetProperties):
        """Create field calculator.

        Parameters
        ----------
        geometry : MotorGeometry
            Layered radii geometry. For thickness-based construction use
            MotorGeometry.from_thicknesses(...).
        magnet : MagnetProperties
            Magnet material and arc description.
        """
        self.geometry = geometry
        self.magnet = magnet
        self.mu0 = 4 * npy.pi * 1e-7  # Permeability of free space
        
        # Calculate derived parameters
        self._calculate_derived_parameters()
    
    def _calculate_derived_parameters(self):
        """
        Calculate derived geometric parameters.
        
        Following the MATLAB model geometry convention:
        - Rs: Stator inner radius
        Updated / Clarified convention (thickness-based construction):
        Outward increasing radii order:
            rotor_inner_radius < rotor_back_iron_outer (= magnet inner) < magnet outer < stator inner < stator outer

        We expose both magnet inner and outer radii explicitly to remove ambiguity:
            Rm_in  = magnet inner radius (outer surface of rotor back-iron)
            Rm_out = magnet outer radius (inner boundary of airgap)
        The airgap spans Rm_out .. Rs.
        """
        self.pole_pitch = npy.pi / self.geometry.pole_pairs
        self.magnet_arc = self.magnet.magnet_arc_ratio * self.pole_pitch
        
    # Consistent physical convention (outward radius increases):
    # Rs: stator inner radius (airgap outer boundary)
    # Rg_in: airgap inner boundary (magnet outer surface) = Rm_out
    # Rm_in: magnet inner boundary (rotor back-iron outer radius) = Rr
    # Rr: rotor back-iron outer boundary (under magnets)
    # NOTE: We expose Zhu-style aliases so the closed-form equations can
    #       use their native variable symbols without rewriting formulas.
        self.Rs = self.geometry.stator_inner_radius
        self.g = self.geometry.airgap_length
        self.Hm = self.magnet.magnet_thickness
        self.coil_thickness = getattr(self.geometry, "coil_thickness", 0.0)
        # Effective outer free-space boundary including coil region
        self.Rs_eff = self.Rs + self.coil_thickness

        # Geometry provided in MotorGeometry uses rotor_outer_radius as airgap inner (magnet outer)
        Rg_in = self.geometry.rotor_outer_radius
        self.Rm_out = Rg_in                 # Magnet outer radius (airgap inner boundary)
        self.Rm_in = Rg_in - self.Hm        # Magnet inner radius (rotor back-iron outer)
        self.Rr = self.Rm_in                # Back-iron outer radius (kept for backward compatibility)

        # Backward compatible names (previous code used self.Rm as magnet outer radius)
        self.Rm = self.Rm_out

        # Relative permeances (for potential future use)
        self.kr = self.geometry.rotor_outer_radius / self.geometry.stator_inner_radius
        self.kg = self.geometry.airgap_length / self.geometry.stator_inner_radius
    
    def fourier_coefficients(self, n_harmonics: int = 50) -> Tuple[npy.ndarray, npy.ndarray]:
        """
        Calculate Fourier coefficients for the magnetization distribution.
        
        Based on Zhu Part 1, Equation (9):
        Mn = (4*Br)/(π*μ0) * sin(α*n)/n
        
        Args:
            n_harmonics: Number of harmonics to calculate
            
        Returns:
            Tuple of (harmonic_orders, magnetization_coefficients)
        """
        p = self.geometry.pole_pairs
        
        # Generate harmonic orders: 1, 3, 5, 7, ... (odd harmonics only)
        harmonic_multipliers = npy.arange(1, 2*n_harmonics, 2)  # [1, 3, 5, 7, ...]
        
        magnetization_coeffs = npy.zeros(len(harmonic_multipliers))
        
        # Magnetization calculation
        M_ = 4 * self.magnet.residual_flux_density / (npy.pi * self.mu0)
        alpha_ = npy.pi * self.magnet.magnet_arc_ratio / 2
        
        for i in range(len(harmonic_multipliers)):
            n = harmonic_multipliers[i]  # Harmonic order: 1, 3, 5, 7, ...
            
            # Magnetization coefficient: M = M_ * sin(alpha_ * n) / n
            magnetization_coeffs[i] = M_ * npy.sin(alpha_ * n) / n
        
        # Return harmonic orders and magnetization coefficients
        harmonic_orders = harmonic_multipliers * p  # [p, 3p, 5p, 7p, ...]
        
        return harmonic_orders, magnetization_coeffs
    
    def radial_flux_density(self, radius: npy.ndarray, theta: npy.ndarray,
                           n_harmonics: int = 20,
                           *,
                           extend_with_coil: bool = False,
                           coil_thickness: float = 0.0) -> npy.ndarray:
        """
        Calculate radial flux density in the airgap.
        
        Based on Zhu Part 1, Equation (34):
        Br(r,θ) = Σ Bn * fn(r) * cos(nθ)
        
        Where:
        - Bn: Flux density coefficients calculated from magnetization Mn
        - fn(r): Radial variation function in airgap
        - n: Harmonic order (pole-pair harmonics: p, 3p, 5p, ...)
        
        Args:
            radius: Radial positions (m)
            theta: Angular positions (rad)
            n_harmonics: Number of harmonics to include
            extend_with_coil: When True, treat a slotless coil layer (μ≈μ0)
                as part of the air region, extending the outer boundary from
                Rs to Rs+coil_thickness.
            coil_thickness: Thickness of the slotless coil layer (m). Only
                used when extend_with_coil=True.
            
        Returns:
            Radial flux density Br(r,θ) (T)
        """
        harmonics, magnetization_coeffs = self.fourier_coefficients(n_harmonics)

        # Calculate flux density coefficients from magnetization (legacy formulation)
        flux_coeffs = npy.zeros(len(magnetization_coeffs))

        # Physical stator inner surface
        Rs_physical = self.geometry.stator_inner_radius
        # Effective outer boundary if coil treated as air
        Rs_eff = Rs_physical + (coil_thickness if extend_with_coil else 0.0)
        if Rs_eff <= self.geometry.rotor_outer_radius:  # safety fallback
            Rs_eff = Rs_physical

        # Legacy coefficient parameters (keep Zhu derivation intact)
        # Corrected geometry: airgap is INSIDE the stator, not outside
        Rs = Rs_physical
        g = self.geometry.airgap_length
        Hm = self.magnet.magnet_thickness
        Rm = Rs - g  # Magnet outer = stator inner - airgap (correct: inward from Rs)
        Rr = self.geometry.rotor_outer_radius  # Magnet inner = magnet outer - magnet thickness
        mur = self.magnet.relative_permeability
        p = self.geometry.pole_pairs

        for i in range(len(magnetization_coeffs)):
            n = 2*i + 1
            n_p = n * p
            M = magnetization_coeffs[i]
            if i == 0:
                numerator = (Rm/Rs)**2 - (Rr/Rs)**2 + (Rr/Rs)**2 * npy.log((Rm/Rr)**2)
                denominator = ((mur+1)/mur)*(1-(Rr/Rs)**2) - ((mur-1)/mur)*((Rm/Rs)**2 - (Rr/Rm)**2)
                flux_coeffs[i] = (self.mu0*M/mur) * numerator / denominator
            else:
                factor1 = 2*(self.mu0*M/mur)
                factor2 = n_p/(n_p**2 - 1)
                factor3 = (Rs/Rm)**(n_p+1)
                numerator = (n_p - 1) + 2*(Rr/Rm)**(n_p+1) - (n_p + 1)*(Rr/Rm)**(2*n_p)
                denominator = ((mur+1)/mur)*(1-(Rr/Rs)**(2*n_p)) - ((mur-1)/mur)*((Rm/Rs)**(2*n_p) - (Rr/Rm)**(2*n_p))
                flux_coeffs[i] = factor1 * factor2 * factor3 * numerator / denominator

        R, THETA = npy.meshgrid(radius, theta, indexing='ij')
        Br = npy.zeros_like(R)

        for n, coeff in zip(harmonics, flux_coeffs):
            if coeff != 0:
                mask = (R >= Rr) & (R <= Rs_eff)
                if not npy.any(mask):
                    continue
                r_factor = npy.zeros_like(R)
                r_ratio = R[mask] / Rs_eff
                rr_rs_ratio = Rr / Rs_eff
                r_factor[mask] = ((r_ratio)**(n-1) - (rr_rs_ratio)**(2*n) * (r_ratio)**(-n-1)) / (1 - (rr_rs_ratio)**(2*n))
                Br += coeff * r_factor * npy.cos(n * THETA)

        return Br

    def radial_flux_density_extended(self, radius: npy.ndarray, theta: npy.ndarray,
                                     n_harmonics: int = 20) -> npy.ndarray:
        """Extended radial flux density treating coil region as part of the airgap.

        This adapts Equation (34) by replacing the outer boundary Rs with Rs_eff = Rs + coil_thickness
        (coil copper + former assumed μ≈μ0). The solution re-derives coefficients using Rs_eff so the
        Laplace solution spans rotor_outer_radius .. Rs_eff.

        Parameters
        ----------
        radius : ndarray
            Radial positions at which to evaluate (can include values up to Rs_eff).
        theta : ndarray
            Angular positions (rad).
        n_harmonics : int
            Number of odd harmonics to include.

        Returns
        -------
        ndarray
            Br(r,θ) over requested grid.
        """
        if self.coil_thickness <= 0:
            # Fall back to original implementation
            return self.radial_flux_density(radius, theta, n_harmonics)

        harmonics, magnetization_coeffs = self.fourier_coefficients(n_harmonics)
        flux_coeffs = npy.zeros(len(magnetization_coeffs))

        # Geometry with extended outer boundary
        Rs = self.Rs          # coil outer / yoke inner (effective boundary)
        g = self.g
        Hm = self.Hm
        r = 0
        # Magnet outer stays at Rs_inner + g (mechanical airgap ends at coil inner)
        Rm = Rs - g
        Rr = Rs - g - Hm
        mur = self.magnet.relative_permeability
        p = self.geometry.pole_pairs

        # Coefficient derivation replaces Rs with Rs_eff in boundary terms; ratios using Rs_eff.
        for i in range(len(magnetization_coeffs)):
            n = 2*i + 1
            n_p = n * p
            M = magnetization_coeffs[i]

            if i == 0:
                numerator = (Rm/Rs)**2 - (Rr/Rs)**2 + (Rr/Rs)**2 * npy.log((Rm/Rr)**2)
                denominator = ((mur+1)/mur)*(1-(Rr/Rs)**2) - ((mur-1)/mur)*((Rm/Rs)**2 - (Rr/Rm)**2)
                flux_coeffs[i] = (self.mu0*M/mur) * numerator / denominator
            else:
                factor1 = 2*(self.mu0*M/mur)
                factor2 = n_p/(n_p**2-1)
                factor3 = (r/Rs)**(n_p - 1)*(Rm/Rs)**(n_p+1) + (Rm/r)**(n_p + 1)
                numerator = (n_p-1) + 2*(Rr/Rm)**(n_p+1) - (n_p+1)*(Rr/Rm)**(2*n_p)
                denominator = ((mur+1)/mur)*(1-(Rr/Rs)**(2*n_p)) - ((mur-1)/mur)*((Rm/Rs)**(2*n_p)-(Rr/Rm)**(2*n_p))
                flux_coeffs[i] = factor1 * factor2 * factor3 * numerator / denominator

        R, THETA = npy.meshgrid(radius, theta, indexing='ij')
        Br = npy.zeros_like(R)

        for n, coeff in zip(harmonics, flux_coeffs):
            if coeff != 0:
                mask = (R >= self.geometry.rotor_outer_radius) & (R <= Rs)
                if npy.any(mask):
                    r_ratio = R[mask] / Rs
                    rr_rs_ratio = self.geometry.rotor_outer_radius / Rs
                    Br_term = ((r_ratio)**(n-1) - (rr_rs_ratio)**(2*n) * (r_ratio)**(-n-1)) / (1 - (rr_rs_ratio)**(2*n))
                    Br += coeff * Br_term * npy.cos(n * THETA)

        return Br


    def radial_flux_density_in_gap_at_radius(self, r: float, theta: npy.ndarray,
                                     n_harmonics: int = 20) -> npy.ndarray:
        """

        Parameters
        ----------
        r : float
            Radial position at which to evaluate (should be within the airgap region).
        theta : ndarray
            Angular positions (rad).
        n_harmonics : int
            Number of odd harmonics to include.

        Returns
        -------
        ndarray
            Br(r,θ) over requested grid.
        """

        harmonics, magnetization_coeffs = self.fourier_coefficients(n_harmonics)
        flux_coeffs = npy.zeros(len(magnetization_coeffs))

        # Geometry with extended outer boundary
        Rs = self.Rs
        g = self.g
        Hm = self.Hm
        # Magnet outer stays at Rs_inner + g (mechanical airgap ends at coil inner)
        Rm = Rs - g
        Rr = Rs - g - Hm
        mur = self.magnet.relative_permeability
        p = self.geometry.pole_pairs

        if r < Rm or r > Rs:
            print ("Requested radius outside airgap region.")
            # Fall back to original implementation
            return any

        Br = npy.zeros_like(theta)

        for i in range(len(magnetization_coeffs)):
            n = 2*i + 1
            n_p = n * p
            M = magnetization_coeffs[i]

            # if i == 0:
            #     numerator = (Rm/Rs)**2 - (Rr/Rs)**2 + (Rr/Rs)**2 * npy.log((Rm/Rr)**2)
            #     denominator = ((mur+1)/mur)*(1-(Rr/Rs)**2) - ((mur-1)/mur)*((Rm/Rs)**2 - (Rr/Rm)**2)
            #     flux_coeffs[i] = (self.mu0*M/mur) * numerator / denominator
            # else:
            factor1 = (self.mu0*M/mur)
            factor2 = n_p/(n_p**2-1)
            factor3 = (r/Rs)**(n_p - 1)*(Rm/Rs)**(n_p+1) + (Rm/r)**(n_p + 1)
            numerator = (n_p-1) + 2*(Rr/Rm)**(n_p+1) - (n_p+1)*(Rr/Rm)**(2*n_p)
            denominator = ((mur+1)/mur)*(1-(Rr/Rs)**(2*n_p)) - ((mur-1)/mur)*((Rm/Rs)**(2*n_p)-(Rr/Rm)**(2*n_p))
            flux_coeffs[i] = factor1 * factor2 * factor3 * numerator / denominator
            print(f"n={n}, Mn={M:.6e}, Bn={flux_coeffs[i]:.6e}")


        for n, coeff in zip(harmonics, flux_coeffs):
            Br += coeff * npy.cos(n * theta)

        return Br


    def radial_flux_density_in_magnets(self, radius: npy.ndarray, theta: npy.ndarray, 
                                      n_harmonics: int = 20) -> npy.ndarray:
        """
        Calculate radial flux density inside the permanent magnets.
        
        Based on Zhu Part 1, Equation (36):
        Complete implementation matching Zhu's exact formulation
        
        Args:
            radius: Radial positions inside magnets (m) - must be Rm ≤ r ≤ Rr
            theta: Angular positions (rad)
            n_harmonics: Number of harmonics to include
            
        Returns:
            Radial flux density inside magnets Br_magnet(r,θ) (T)
        """
        harmonics, _ = self.fourier_coefficients(n_harmonics)
        
        R, THETA = npy.meshgrid(radius, theta, indexing='ij')
        Br_magnet = npy.zeros_like(R)
        
        # Geometry parameters from Zhu's convention
        Rs = self.Rs
        Rm = self.Rm  # Magnet inner surface
        Rr = self.Rr  # Magnet outer surface (rotor surface)
        mur = self.magnet.relative_permeability
        
        # Base magnetization coefficient
        M_ = 4 * self.magnet.residual_flux_density / (npy.pi * self.mu0)
        alpha_ = npy.pi * self.magnet.magnet_arc_ratio / 2
        p = self.geometry.pole_pairs
        
        for i, n_harmonic in enumerate(harmonics):
            n = (2*i + 1)  # Actual harmonic order: 1, 3, 5, 7, ...
            n_p = n * p     # MATLAB variable name
            
            # Magnetization coefficient for this harmonic
            if npy.sin(alpha_ * n) != 0:
                Mn = M_ * npy.sin(alpha_ * n) / n
                
                # Only calculate for points inside magnets
                mask = (R >= Rm) & (R <= Rr)
                
                if npy.any(mask):
                    r = R[mask]
                    
                    # Zhu Equation 36 - First term (complex boundary condition term)
                    numerator1 = (n_p - 1/mur) * Rm**(2*n_p) + (1 + 1/mur) * Rr**(n_p+1) * Rm**(n_p-1) - (n_p + 1/mur) * Rs**(2*n_p) - (1 - 1/mur) * (Rr/Rm)**n_p * Rs**(2*n_p)
                    denominator1 = ((mur + 1)/mur) * (Rs**(2*n_p) - Rr**(2*n_p)) - ((mur - 1)/mur) * (Rm**(2*n_p) - Rs**(2*n_p) * (Rr/Rm)**(2*n_p))
                    term1_coeff = self.mu0 * Mn * n_p / ((n_p)**2 - 1) * Rm**(-n_p-1)
                    term1 = term1_coeff * (numerator1 / denominator1) * (r**(n_p-1) + Rr**(2*n_p) * r**(-n_p-1))
                    
                    # Zhu Equation 36 - Second term
                    term2 = self.mu0 * Mn * n_p / ((n_p)**2 - 1) * (Rr/r)**(n_p+1)
                    
                    # Zhu Equation 36 - Third term
                    term3 = self.mu0 * Mn * (n_p)**2 / ((n_p)**2 - 1)
                    
                    # Complete Zhu Equation 36
                    Br_harmonic = npy.zeros_like(R)
                    Br_harmonic[mask] = (term1 + term2 + term3) * npy.cos(n_p * THETA[mask])
                    
                    Br_magnet += Br_harmonic
        
        return Br_magnet
    
    def tangential_flux_density(self, radius: npy.ndarray, theta: npy.ndarray,
                               n_harmonics: int = 20) -> npy.ndarray:
        """
        Calculate tangential flux density in the airgap.
        
        Based on Zhu Part 1, Equation (35):
        Bt(r,θ) = Σ Bn * gn(r) * sin(nθ)
        
        Where:
        - Bn: Flux density coefficients (same as for radial component)
        - gn(r): Radial variation function for tangential component
        - n: Harmonic order (pole-pair harmonics: p, 3p, 5p, ...)
        
        Args:
            radius: Radial positions (m)
            theta: Angular positions (rad)  
            n_harmonics: Number of harmonics to include
            
        Returns:
            Tangential flux density Bt(r,θ) (T)
        """
        harmonics, coeffs = self.fourier_coefficients(n_harmonics)
        
        R, THETA = npy.meshgrid(radius, theta, indexing='ij')
        Bt = npy.zeros_like(R)
        
        Rs = self.geometry.stator_inner_radius
        Rr = self.geometry.rotor_outer_radius
        
        for n, coeff in zip(harmonics, coeffs):
            if coeff != 0:
                # Only calculate for points in the airgap
                mask = (R >= Rr) & (R <= Rs)
                r_factor = npy.zeros_like(R)
                
                if npy.any(mask):
                    r_ratio = R[mask] / Rs
                    rr_rs_ratio = Rr / Rs
                    
                    # Zhu's radial variation function gn(r) - Equation (18)
                    # gn(r) = [(r/Rs)^(n-1) + (Rr/Rs)^(2n) * (r/Rs)^(-n-1)] / [1 - (Rr/Rs)^(2n)]
                    r_factor[mask] = ((r_ratio)**(n-1) + (rr_rs_ratio)**(2*n) * (r_ratio)**(-n-1)) / \
                                   (1 - (rr_rs_ratio)**(2*n))
                    
                    # Zhu Equation (35): Bt = Σ Bn * gn(r) * sin(nθ)
                    # Note: The relationship between Bt and Br comes from ∇×B=0
                    Bt += coeff * r_factor * npy.sin(n * THETA)
        
        return Bt
    
    def tangential_flux_density_in_magnets(self, radius: npy.ndarray, theta: npy.ndarray,
                                          n_harmonics: int = 20) -> npy.ndarray:
        """
        Calculate tangential flux density inside the permanent magnets.
        
        Based on Zhu Part 1, equations for tangential field in magnet region:
        Bt_magnet(r,θ) = Σ [An * (n-1) * (r/Rs)^(n-2) - Bn * (n+1) * (r/Rs)^(-n-2)] * sin(nθ)
        
        Where An, Bn are the same coefficients as in radial field calculation.
        
        Complete formulation from Zhu without simplifications:
        - Derived from Bt = -(1/r) * ∂Φ/∂θ
        - Uses same boundary conditions as radial component
        - Accounts for magnetization source terms
        
        Args:
            radius: Radial positions inside magnets (m) - must be Rm ≤ r ≤ Rr
            theta: Angular positions (rad)
            n_harmonics: Number of harmonics to include
            
        Returns:
            Tangential flux density inside magnets Bt_magnet(r,θ) (T)
        """
        harmonics, _ = self.fourier_coefficients(n_harmonics)
        
        R, THETA = np.meshgrid(radius, theta, indexing='ij')
        Bt_magnet = np.zeros_like(R)
        
        # Geometry parameters (your MATLAB convention)
        Rs = self.Rs
        Rm = self.Rm  # Magnet inner surface
        Rr = self.Rr  # Magnet outer surface (rotor surface)
        mur = self.magnet.relative_permeability
        
        # Base magnetization coefficient
        M_ = 4 * self.magnet.residual_flux_density / (np.pi * self.mu0)
        alpha_ = np.pi * self.magnet.magnet_arc_ratio / 2
        p = self.geometry.pole_pairs
        
        for i, n_harmonic in enumerate(harmonics):
            n = (2*i + 1)  # Actual harmonic order: 1, 3, 5, 7, ...
            n_p = n * p     # MATLAB variable name
            
            # Magnetization coefficient for this harmonic
            if np.sin(alpha_ * n) != 0:
                Mn = M_ * np.sin(alpha_ * n) / n
                
                # Only calculate for points inside magnets
                mask = (R >= Rm) & (R <= Rr)
                
                if np.any(mask):
                    r_ratio = R[mask] / Rs
                    rm_ratio = Rm / Rs
                    rr_ratio = Rr / Rs
                    
                    if n == 1:  # Special case for fundamental (n=1)
                        # For n=1: Bt = -∂Φ/∂θ where Φ has logarithmic terms
                        
                        source_term = (self.mu0 * Mn) / mur
                        
                        # Derivative of particular solution for n=1
                        particular_deriv_numerator = -source_term * (r_ratio**2 - rm_ratio**2) * p
                        particular_deriv_denominator = 2
                        particular_deriv = particular_deriv_numerator / particular_deriv_denominator
                        
                        # Homogeneous solution derivative (A1 coefficient from radial calculation)
                        A1_numerator = source_term * rm_ratio**2 * np.log(rr_ratio/rm_ratio)
                        A1_denominator = 2 * np.log(rr_ratio/rm_ratio)
                        A1_coeff = A1_numerator / A1_denominator
                        homogeneous_deriv = -A1_coeff * p
                        
                        # Complete n=1 tangential solution inside magnet
                        Bt_harmonic = np.zeros_like(R)
                        Bt_harmonic[mask] = (particular_deriv + homogeneous_deriv) * np.sin(n_p * THETA[mask])
                        
                    else:  # n > 1 harmonics - Zhu Equation 37
                        # Zhu Equation 37 - Tangential flux density for n_p != 1
                        
                        r = R[mask]
                        
                        # First term - complex boundary condition term
                        numerator1 = ((n_p - 1)/mur) * Rm**(2*n_p) + (1 + 1/mur) * Rr**n_p * Rm**(n_p-1) - ((n_p + 1)/mur) * Rs**(2*n_p) - (1 - 1/mur) * (Rr/Rs)**n_p * Rs**(2*n_p)
                        denominator1 = ((mur + 1)/mur) * (Rs**(2*n_p) - Rr**(2*n_p)) - ((mur - 1)/mur) * (Rm**(2*n_p) - Rs**(2*n_p) * (Rr/Rm)**(2*n_p))
                        term1_coeff = (-self.mu0 * Mn) * n_p / ((n_p)**2 - 1) * Rm**(-n_p-1)
                        term1 = term1_coeff * (numerator1 / denominator1) * (r**(n_p-1) - Rr**(2*n_p) * r**(-n_p-1))
                        
                        # Second term
                        term2 = self.mu0 * Mn * n_p / ((n_p)**2 - 1) * (Rr/r)**(n_p+1)
                        
                        # Third term  
                        term3 = -self.mu0 * Mn * n_p / ((n_p)**2 - 1)
                        
                        # Complete Zhu Equation 37
                        Bt_harmonic = np.zeros_like(R)
                        Bt_harmonic[mask] = (term1 + term2 + term3) * np.sin(n_p * THETA[mask])
                    
                    Bt_magnet += Bt_harmonic
        
        return Bt_magnet
    
    def magnetization_distribution(self, radius: np.ndarray, theta: np.ndarray,
                                  n_harmonics: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the magnetization distribution inside permanent magnets.
        
        Based on Zhu Part 1, the magnetization has both radial and tangential components:
        Mr(r,θ) = Σ Mn * cos(nθ)  - Radial magnetization
        Mt(r,θ) = 0               - Tangential magnetization (radially magnetized)
        
        Where Mn are the Fourier coefficients of the magnetization pattern.
        
        Complete formulation without simplifications:
        - Mr is independent of radius (uniform magnetization through magnet thickness)
        - Pattern follows the magnet arc ratio and pole arrangement
        - Includes all harmonic components
        
        Args:
            radius: Radial positions inside magnets (m)
            theta: Angular positions (rad)
            n_harmonics: Number of harmonics to include
            
        Returns:
            Tuple of (Mr, Mt) - radial and tangential magnetization (A/m)
        """
        R, THETA = np.meshgrid(radius, theta, indexing='ij')
        Mr = np.zeros_like(R)
        Mt = np.zeros_like(R)  # Zero for radially magnetized magnets
        
        # Base magnetization amplitude
        M_base = 4 * self.magnet.residual_flux_density / (np.pi * self.mu0)
        alpha_ = np.pi * self.magnet.magnet_arc_ratio / 2
        p = self.geometry.pole_pairs
        
        # Only calculate for points inside magnets
        mask = (R >= self.Rm) & (R <= self.Rr)
        
        for i in range(n_harmonics):
            n = 2*i + 1  # Odd harmonics: 1, 3, 5, 7, ...
            n_p = n * p   # MATLAB variable name
            
            # Magnetization coefficient for this harmonic
            if np.sin(alpha_ * n) != 0:
                Mn = M_base * np.sin(alpha_ * n) / n
                
                # Add harmonic contribution (uniform in radius)
                Mr[mask] += Mn * np.cos(n_p * THETA[mask])
        
        return Mr, Mt
    
    def magnetic_field_strength_in_magnets(self, radius: np.ndarray, theta: np.ndarray,
                                         n_harmonics: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate magnetic field strength H inside permanent magnets.
        
        Based on Zhu Part 1, using the relationship:
        B = μ0 * μr * H + μ0 * M
        
        Therefore: H = (B - μ0 * M) / (μ0 * μr)
        
        Complete formulation without simplifications:
        - Accounts for finite permeability of magnets
        - Includes demagnetization effects
        - Uses complete magnetization distribution
        
        Args:
            radius: Radial positions inside magnets (m)
            theta: Angular positions (rad)
            n_harmonics: Number of harmonics to include
            
        Returns:
            Tuple of (Hr, Ht) - radial and tangential H-field (A/m)
        """
        # Get flux density inside magnets
        Br_magnet = self.radial_flux_density_in_magnets(radius, theta, n_harmonics)
        Bt_magnet = self.tangential_flux_density_in_magnets(radius, theta, n_harmonics)
        
        # Get magnetization distribution
        Mr, Mt = self.magnetization_distribution(radius, theta, n_harmonics)
        
        # Calculate H-field using B = μ0*μr*H + μ0*M
        mur = self.magnet.relative_permeability
        
        Hr = (Br_magnet - self.mu0 * Mr) / (self.mu0 * mur)
        Ht = (Bt_magnet - self.mu0 * Mt) / (self.mu0 * mur)
        
        return Hr, Ht
    
    def radial_flux_density_equation_52(self, theta: np.ndarray, 
                                       n_harmonics: int = 20) -> np.ndarray:
        """
        Calculate radial flux density BH(θ) at the stator surface using Zhu's Equation 52.
        
        Based on Zhu Part 1, Equation (52):
        BH(θ) = Σ 2 * (μ0*Mn)/(μr) * (n_p)/((n_p)²-1) * (Rs/Rm)^(n_p-1)
                * {[(n_p-1)*Rm^(2*n_p) + 2*Rs^(n_p+1)*Rm^(n_p-1) - (n_p+1)*Rs^(2*n_p)] / 
                   [μr+1)/μr * [Rs^(2*n_p) - Rm^(2*n_p)] - (μr-1)/μr * [Rm^(2*n_p) - Rs^(2*n_p)*(Rs/Rm)^(2*n_p)]]}
                * cos(n_p*θ)
        
        This equation provides the radial flux density at the stator inner surface,
        considering complete boundary conditions and magnet permeability effects.
        
        Geometry convention (Zhu's notation):
        - Rs: Stator inner radius (airgap outer boundary) - evaluation point
        - Rm: Magnet inner radius (rotor back-iron outer radius)
        - Physical arrangement: Rm < Rr < Rs (rotor < magnet surface < stator)
        
        Note: Summation is over n = 1, 3, 5, ... where n_p = n*p (pole-pair harmonic order)
        
        Args:
            theta: Angular positions (rad)
            n_harmonics: Number of odd harmonics to include (1, 3, 5, ...)
            
        Returns:
            BH: Radial flux density at stator surface (T)
        """
        # Initialize output array
        BH = np.zeros_like(theta)
        
        # Motor parameters - use Zhu's convention for Equation 52
        # In Zhu Eq 52: Rs = stator inner (airgap outer), Rm = magnet inner radius
        # Physical: rotor_back_iron < Rm < Rr (rotor outer with magnets) < Rs (stator inner)
        Rs = self.geometry.stator_inner_radius  # Stator inner radius (evaluation surface)
        # Use clarified attribute for magnet inner radius (Equation 52 uses magnet inner)
        Rm = self.Rm_in
        mur = self.magnet.relative_permeability
        p = self.geometry.pole_pairs
        
        # Base magnetization coefficient
        M_ = 4 * self.magnet.residual_flux_density / (np.pi * self.mu0)
        alpha_ = np.pi * self.magnet.magnet_arc_ratio / 2
        
        # Calculate each harmonic contribution
        # Summation over n = 1, 3, 5, 7, ... (odd harmonics)
        for i in range(n_harmonics):
            n = 2*i + 1  # Harmonic multiplier: 1, 3, 5, 7, ...
            n_p = n * p  # Pole-pair harmonic order
            
            # Calculate magnetization coefficient for this harmonic
            if np.sin(alpha_ * n) == 0:
                continue
                
            Mn = M_ * np.sin(alpha_ * n) / n
            
            if n_p == 1:
                # Skip n_p=1 case as it causes division by zero in (n_p²-1)
                continue
                
            # Coefficient term: 2 * (μ0*Mn)/μr * n_p/((n_p)²-1) * (Rs/Rm)^(n_p-1)
            coeff = (2 * self.mu0 * Mn / mur * 
                    n_p / (n_p**2 - 1) * 
                    (Rs / Rm)**(n_p - 1))
            
            # Numerator of the complex fraction
            numerator = ((n_p - 1) * Rm**(2*n_p) + 
                        2 * Rs**(n_p + 1) * Rm**(n_p - 1) - 
                        (n_p + 1) * Rs**(2*n_p))
            
            # Denominator of the complex fraction
            term1 = ((mur + 1) / mur) * (Rs**(2*n_p) - Rm**(2*n_p))
            term2 = ((mur - 1) / mur) * (Rm**(2*n_p) - Rs**(2*n_p) * (Rs/Rm)**(2*n_p))
            denominator = term1 - term2
            
            # Check for zero or invalid denominator (numerical stability)
            # Note: denominator can be very small due to high powers of Rs, Rm < 1
            # Only skip if exactly zero or would cause overflow
            if denominator == 0 or not np.isfinite(denominator):
                continue
            
            # Complete harmonic term
            harmonic_term = coeff * (numerator / denominator) * np.cos(n_p * theta)
            
            # Add to total
            BH += harmonic_term
        
        return BH
    
    def flux_linkage_per_turn(self, radius: float, n_harmonics: int = 20) -> float:
        """
        Calculate flux linkage per turn at given radius.
        
        Physical note: Directly integrating Br over a full pole pair cancels to ~0
        for sinusoidal fields. The useful linkage is the fundamental flux per pole
        that links a full-pitch turn. We therefore extract the fundamental cosine
        component of Br at the given radius and compute:
            Φ_pole = (2/p) * B1(R, fundamental) * R * L
        where B1 is the amplitude of the fundamental pole-pair harmonic at the
        stator surface.

        Args:
            radius: Radius at which to calculate flux linkage (m)
            n_harmonics: Number of odd harmonics to include in Br synthesis

        Returns:
            Flux linkage per turn (Wb), equal to the fundamental flux per pole
            that a full-pitch single turn links.
        """
        # Sample Br over 0..2π and project onto the fundamental cos(pθ)
        p = self.geometry.pole_pairs
        theta = np.linspace(0.0, 2.0*np.pi, 2048, endpoint=False)
        Br = self.radial_flux_density_in_gap_at_radius(radius, theta, n_harmonics)[0, :]

        # Fundamental amplitude at this radius using orthogonality:
        # A1 = (1/π) ∫_0^{2π} Br(θ) cos(pθ) dθ
        cos_p = np.cos(p * theta)
        A1 = (1.0/np.pi) * np.trapz(Br * cos_p, theta)

        # Fundamental flux per pole linking a full-pitch turn
        phi_per_pole = (2.0 / p) * A1 * radius * self.geometry.axial_length
        return float(phi_per_pole)
    
    def plot_flux_density_distribution(self, radius: Optional[float] = None,
                                     n_harmonics: int = 20, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot the flux density distribution in the airgap.
        
        Visualizes the results from Zhu Part 1, Equations (15-18):
        - Radial flux density Br(r,θ)
        - Tangential flux density Bt(r,θ)
        
        Args:
            radius: Radius at which to plot (default: middle of airgap)
            n_harmonics: Number of harmonics to include
            figsize: Figure size
        """
        if radius is None:
            radius = (self.geometry.rotor_outer_radius + self.geometry.stator_inner_radius) / 2
        
        theta_deg = np.linspace(0, 360, 720)
        theta_rad = np.deg2rad(theta_deg)
        
        Br = self.radial_flux_density(np.array([radius]), theta_rad, n_harmonics)
        Bt = self.tangential_flux_density(np.array([radius]), theta_rad, n_harmonics)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Radial flux density
        ax1.plot(theta_deg, Br[0, :], 'b-', linewidth=2, label='Radial flux density')
        ax1.set_xlabel('Angular position (degrees)')
        ax1.set_ylabel('Radial flux density (T)')
        ax1.set_title(f'Open-Circuit Radial Flux Density at r = {radius*1000:.1f} mm')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Tangential flux density  
        ax2.plot(theta_deg, Bt[0, :], 'r-', linewidth=2, label='Tangential flux density')
        ax2.set_xlabel('Angular position (degrees)')
        ax2.set_ylabel('Tangential flux density (T)')
        ax2.set_title(f'Open-Circuit Tangential Flux Density at r = {radius*1000:.1f} mm')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
    def plot_complete_field_distribution(self, n_harmonics: int = 20, 
                                        figsize: Tuple[int, int] = (16, 12)):
        """
        Plot complete magnetic field distribution in all regions.
        
        Shows the field in:
        1. Airgap region (Rs ≤ r ≤ Rm)
        2. Magnet region (Rm ≤ r ≤ Rr)
        3. Magnetization distribution
        4. H-field inside magnets
        
        Args:
            n_harmonics: Number of harmonics to include
            figsize: Figure size
        """
        theta_deg = np.linspace(0, 360, 720)
        theta_rad = np.deg2rad(theta_deg)
        
        # Define radial positions in different regions
        r_airgap = np.array([(self.Rs + self.Rm) / 2])  # Middle of airgap
        r_magnet = np.array([(self.Rm + self.Rr) / 2])  # Middle of magnet
        
        # Calculate fields in different regions
        Br_airgap = self.radial_flux_density(r_airgap, theta_rad, n_harmonics)
        Bt_airgap = self.tangential_flux_density(r_airgap, theta_rad, n_harmonics)
        
        Br_magnet = self.radial_flux_density_in_magnets(r_magnet, theta_rad, n_harmonics)
        Bt_magnet = self.tangential_flux_density_in_magnets(r_magnet, theta_rad, n_harmonics)
        
        Mr, Mt = self.magnetization_distribution(r_magnet, theta_rad, n_harmonics)
        Hr_magnet, Ht_magnet = self.magnetic_field_strength_in_magnets(r_magnet, theta_rad, n_harmonics)
        
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=figsize)
        
        # Flux density in airgap
        ax1.plot(theta_deg, Br_airgap[0, :], 'b-', linewidth=2, label=f'Br at r={r_airgap[0]*1000:.1f}mm')
        ax1.plot(theta_deg, Bt_airgap[0, :], 'r-', linewidth=2, label=f'Bt at r={r_airgap[0]*1000:.1f}mm')
        ax1.set_xlabel('Angular position (degrees)')
        ax1.set_ylabel('Flux density (T)')
        ax1.set_title('Magnetic Flux Density in Airgap')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Flux density in magnets
        ax2.plot(theta_deg, Br_magnet[0, :], 'b-', linewidth=2, label=f'Br at r={r_magnet[0]*1000:.1f}mm')
        ax2.plot(theta_deg, Bt_magnet[0, :], 'r-', linewidth=2, label=f'Bt at r={r_magnet[0]*1000:.1f}mm')
        ax2.set_xlabel('Angular position (degrees)')
        ax2.set_ylabel('Flux density (T)')
        ax2.set_title('Magnetic Flux Density inside Magnets')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Magnetization distribution
        ax3.plot(theta_deg, Mr[0, :]/1000, 'g-', linewidth=2, label='Mr (radial)')
        ax3.plot(theta_deg, Mt[0, :]/1000, 'orange', linewidth=2, label='Mt (tangential)')
        ax3.set_xlabel('Angular position (degrees)')
        ax3.set_ylabel('Magnetization (kA/m)')
        ax3.set_title('Magnetization Distribution in Magnets')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # H-field in magnets
        ax4.plot(theta_deg, Hr_magnet[0, :]/1000, 'purple', linewidth=2, label='Hr')
        ax4.plot(theta_deg, Ht_magnet[0, :]/1000, 'brown', linewidth=2, label='Ht')
        ax4.set_xlabel('Angular position (degrees)')
        ax4.set_ylabel('H-field (kA/m)')
        ax4.set_title('Magnetic Field Strength inside Magnets')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Comparison of flux density in both regions
        ax5.plot(theta_deg, Br_airgap[0, :], 'b-', linewidth=2, label='Br in airgap')
        ax5.plot(theta_deg, Br_magnet[0, :], 'b--', linewidth=2, label='Br in magnet')
        ax5.set_xlabel('Angular position (degrees)')
        ax5.set_ylabel('Radial flux density (T)')
        ax5.set_title('Radial Flux Density Comparison')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Field magnitude comparison
        B_mag_airgap = np.sqrt(Br_airgap[0, :]**2 + Bt_airgap[0, :]**2)
        B_mag_magnet = np.sqrt(Br_magnet[0, :]**2 + Bt_magnet[0, :]**2)
        ax6.plot(theta_deg, B_mag_airgap, 'k-', linewidth=2, label='|B| in airgap')
        ax6.plot(theta_deg, B_mag_magnet, 'k--', linewidth=2, label='|B| in magnet')
        ax6.set_xlabel('Angular position (degrees)')
        ax6.set_ylabel('Flux density magnitude (T)')
        ax6.set_title('Flux Density Magnitude Comparison')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\\nField Analysis Summary:")
        print(f"Geometry: Rs={self.Rs*1000:.1f}mm, Rm={self.Rm*1000:.1f}mm, Rr={self.Rr*1000:.1f}mm")
        print(f"Airgap flux density: Br_max={np.max(np.abs(Br_airgap)):.3f}T")
        print(f"Magnet flux density: Br_max={np.max(np.abs(Br_magnet)):.3f}T")
        print(f"Magnetization: Mr_max={np.max(np.abs(Mr))/1000:.1f}kA/m")
        print(f"H-field in magnet: Hr_max={np.max(np.abs(Hr_magnet))/1000:.1f}kA/m")
    
    def harmonic_analysis(self, radius: Optional[float] = None, 
                         n_harmonics: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform harmonic analysis of the flux density.
        
        Analyzes the harmonic content from the Fourier series expansion
        (Zhu Part 1, Equations 8-9) using FFT to validate analytical results.
        
        The harmonic amplitudes should match the theoretical coefficients Mn
        calculated in fourier_coefficients() function.
        
        Args:
            radius: Radius at which to analyze (default: middle of airgap)
            n_harmonics: Number of harmonics to analyze
            
        Returns:
            Tuple of (harmonic_orders, harmonic_amplitudes)
        """
        if radius is None:
            radius = (self.geometry.rotor_outer_radius + self.geometry.stator_inner_radius) / 2
        
        theta = np.linspace(0, 2*np.pi, 2048)
        Br = self.radial_flux_density(np.array([radius]), theta, n_harmonics)
        
        # FFT analysis
        fft_result = np.fft.fft(Br[0, :])
        harmonic_amplitudes = 2 * np.abs(fft_result[:n_harmonics]) / len(theta)
        harmonic_orders = np.arange(1, n_harmonics + 1)
        
        return harmonic_orders, harmonic_amplitudes[1:n_harmonics+1]


def example_usage():
    """Example usage of the OpenCircuitField class"""
    
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
    
    # Create open circuit field calculator
    oc_field = OpenCircuitField(geometry, magnet)
    
    # Calculate and plot flux density distribution
    oc_field.plot_flux_density_distribution()
    
    # Perform harmonic analysis
    harmonics, amplitudes = oc_field.harmonic_analysis()
    
    print("Harmonic Analysis Results:")
    print("Harmonic Order | Amplitude (T)")
    print("-" * 30)
    for h, amp in zip(harmonics[:10], amplitudes[:10]):
        print(f"{h:12d} | {amp:10.4f}")


if __name__ == "__main__":
    example_usage()