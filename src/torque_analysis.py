"""Torque Analysis Utilities

Provides helper functions to compute electromagnetic torque curves based on
open-circuit flux linkage and winding configuration.

Assumptions:
- Sinusoidal airgap field and phase currents
- Linear materials (no saturation modeling)
- Slotless or effectively distributed winding represented by winding_factor
- Current angle gamma defines torque-producing component (q-axis alignment when gamma=0°)
"""
from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np

# Support execution both as a package module and as loose files on sys.path
try:  # absolute imports when 'src' is on sys.path (not a package)
    from open_circuit_field import OpenCircuitField, MotorGeometry, MagnetProperties
    from armature_reaction_field import WindingConfiguration
except ImportError:  # fallback to relative imports when used as a package
    from .open_circuit_field import OpenCircuitField, MotorGeometry, MagnetProperties
    from .armature_reaction_field import WindingConfiguration


def pm_flux_linkage(open_circuit: OpenCircuitField, winding: WindingConfiguration) -> float:
    """Estimate the permanent magnet flux linkage for ONE PHASE (peak amplitude).

    Definition
    -----------
    Returns the PEAK per-phase PM flux linkage ψ_phase_peak (Wb), computed as:
        ψ_phase_peak = ψ_turn_peak(Rs) * N_turns * k_w

    where ψ_turn_peak(Rs) is the fundamental flux per pole linking a full-pitch
    single turn at the stator inner radius (see open_circuit_field.flux_linkage_per_turn).
    This is the correct quantity to use in the standard PMSM torque equation:
        T = 1.5 * p * ψ_phase_peak * I_q_peak.
    """
    psi_turn_peak = open_circuit.flux_linkage_per_turn(open_circuit.Rs)
    psi_phase_peak = psi_turn_peak * winding.turns_per_coil * winding.winding_factor
    return psi_phase_peak


def electromagnetic_torque(
    psi_phase: float,
    p: int,
    current: float,
    gamma_deg: float = 0.0,
    *,
    current_is_rms: bool = True,
    psi_is_peak: bool = True,
) -> float:
    """Compute electromagnetic torque for one phase flux linkage and phase current.

    Contract
    --------
    - psi_phase: per-phase PM flux linkage. If `psi_is_peak=True` (default), it's a peak (amplitude) value.
                 If `False`, it is RMS and will be converted to peak internally.
    - current: phase current magnitude. If `current_is_rms=True` (default), it's RMS; otherwise peak.
    - Formula used with peak quantities: T = 1.5 * p * ψ_phase_peak * I_q_peak * cos(γ)
      (no extra factor of 3; ψ is already per phase).

    Parameters
    ----------
    p : int
        Pole pairs.
    gamma_deg : float
        Electrical angle between phase current and PM flux (deg). gamma=0 gives max torque.

    Returns
    -------
    float
        Electromagnetic torque (Nm).
    """
    gamma = np.deg2rad(gamma_deg)
    I_peak = current * (np.sqrt(2.0) if current_is_rms else 1.0)
    psi_peak = psi_phase if psi_is_peak else (psi_phase * np.sqrt(2.0))
    return 1.5 * p * psi_peak * I_peak * np.cos(gamma)


def current_from_density(J_A_per_m2: float, winding: WindingConfiguration, geometry: MotorGeometry) -> float:
    """Convert current density (A/m^2) to PEAK phase current.

    IMPORTANT: Phase current is set by the current density in the parallel
    conductors of one phase path, not by the number of series turns. Therefore
    we must NOT multiply by turns_per_coil or pole/slot counts here; those
    affect voltage/flux linkage, not current.

    By default, assume a single parallel strand per phase path. If the
    WindingConfiguration provides an optional attribute `parallel_strands`, it
    will be used to scale the phase current accordingly.
    
    Convention: J is the AMPLITUDE (peak) current density, so the returned
    phase current is also peak (no RMS conversion factor applied).
    """
    parallel_strands = getattr(winding, "parallel_strands", 1)
    # Interpret J as peak/amplitude current density → returns I_phase_peak directly
    I_phase_peak = J_A_per_m2 * winding.conductor_area * parallel_strands
    return I_phase_peak


def torque_curve(
    open_circuit: OpenCircuitField,
    winding: WindingConfiguration,
    J_values_A_per_mm2: Iterable[float],
    gamma_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate torque vs current density curve (amplitude/peak current density input).

    Parameters
    ----------
    open_circuit : OpenCircuitField
        Initialized open-circuit field calculator.
    winding : WindingConfiguration
        Winding configuration dataclass.
    J_values_A_per_mm2 : Iterable[float]
        Current density values in A/mm^2 (AMPLITUDE/PEAK).
    gamma_deg : float
        Current angle; 0 for max torque, 90 for pure d-axis (field weakening).

    Returns
    -------
    (J_mm2_array, torque_array) : np.ndarray, np.ndarray
        Current densities (A/mm^2) and corresponding torque (Nm).
    """
    # Use magnitude of peak per-phase flux linkage
    psi_phase_peak = abs(pm_flux_linkage(open_circuit, winding))
    p = open_circuit.geometry.pole_pairs

    J_mm2_array = np.array(list(J_values_A_per_mm2), dtype=float)
    J_m2_array = J_mm2_array * 1e6  # convert to A/m^2

    torque_values = []
    for J_m2 in J_m2_array:
        I_phase_peak = current_from_density(J_m2, winding, open_circuit.geometry)
        T = electromagnetic_torque(
            psi_phase_peak,
            p,
            I_phase_peak,
            gamma_deg,
            current_is_rms=False,  # J is amplitude → I_phase_peak is peak
            psi_is_peak=True,
        )
        torque_values.append(T)

    return J_mm2_array, np.array(torque_values)


__all__ = [
    "pm_flux_linkage",
    "electromagnetic_torque",
    "current_from_density",
    "torque_curve",
]
