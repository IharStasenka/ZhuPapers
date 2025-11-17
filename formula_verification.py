"""
Formula Verification - Python vs MATLAB Format Comparison

This file shows the Python formulas written in the exact same format as your MATLAB code
for easy verification and comparison.

Your MATLAB Code:
================

For i==1 (first harmonic):
Ba(i)=(mu0*M/mur)*((Rm/Rs)^(2)-(Rr/Rs)^(2)+(Rr/Rs)^(2)*log((Rm/Rr)^(2)))/(((mur+1)/mur)*(1-(Rr/Rs)^(2))-((mur-1)/mur)*((Rm/Rs)^2-(Rr/Rm)^2));

For i>1 (higher harmonics):
Ba(i)=2*(mu0*M/mur)*(np/(np^2-1))*((Rs/Rm)^(np+1))*((np-1)+2*(Rr/Rm)^(np+1)-(np+1)*(Rr/Rm)^(2*np))/(((mur+1)/mur)*(1-(Rr/Rs)^(2*np))-((mur-1)/mur)*((Rm/Rs)^(2*np)-(Rr/Rm)^(2*np)));

Python Equivalent:
==================
"""

import numpy as np

def fourier_coefficients_matlab_format(geometry, magnet, n_harmonics=50):
    """
    Python formulas written in exact MATLAB format for verification
    """
    
    # MATLAB variables (same names)
    mu0 = 4 * np.pi * 1e-7
    mur = magnet.relative_permeability
    Br = magnet.residual_flux_density
    alpha_r = magnet.magnet_arc_ratio
    p = geometry.pole_pairs
    
    # MATLAB geometry
    Rs = geometry.stator_inner_radius
    g = geometry.airgap_length
    Hm = magnet.magnet_thickness
    Rm = Rs + g
    Rr = Rs + Hm + g
    
    # MATLAB magnetization calculation
    M_ = 4 * Br / (np.pi * mu0)
    alpha_ = np.pi * alpha_r / 2
    
    # Results array
    Ba = np.zeros(n_harmonics)
    
    print("Formula Verification:")
    print("====================")
    print(f"mu0 = {mu0}")
    print(f"mur = {mur}")
    print(f"Br = {Br}")
    print(f"Rs = {Rs}")
    print(f"Rm = {Rm}")
    print(f"Rr = {Rr}")
    print(f"p = {p}")
    print(f"M_ = {M_}")
    print(f"alpha_ = {alpha_}")
    print()
    
    for i in range(n_harmonics):
        n = 2*i + 1  # Odd harmonics: 1, 3, 5, 7, ...
        np = n * p   # MATLAB variable name
        
        # MATLAB: M = M_ * sin(alpha_ * n) / n
        M = M_ * np.sin(alpha_ * n) / n
        
        if i == 0:  # First harmonic (i==1 in MATLAB)
            print(f"For i=1 (n={n}, np={np}):")
            print("MATLAB Formula:")
            print("Ba(i)=(mu0*M/mur)*((Rm/Rs)^(2)-(Rr/Rs)^(2)+(Rr/Rs)^(2)*log((Rm/Rr)^(2)))/(((mur+1)/mur)*(1-(Rr/Rs)^(2))-((mur-1)/mur)*((Rm/Rs)^2-(Rr/Rm)^2));")
            print()
            print("Python equivalent:")
            
            # Numerator
            numerator = (Rm/Rs)**2 - (Rr/Rs)**2 + (Rr/Rs)**2 * np.log((Rm/Rr)**2)
            print(f"numerator = (Rm/Rs)^2 - (Rr/Rs)^2 + (Rr/Rs)^2 * log((Rm/Rr)^2)")
            print(f"numerator = ({Rm/Rs:.6f})^2 - ({Rr/Rs:.6f})^2 + ({Rr/Rs:.6f})^2 * log(({Rm/Rr:.6f})^2)")
            print(f"numerator = {numerator:.8f}")
            
            # Denominator
            denominator = ((mur+1)/mur) * (1-(Rr/Rs)**2) - ((mur-1)/mur) * ((Rm/Rs)**2 - (Rr/Rm)**2)
            print(f"denominator = ((mur+1)/mur) * (1-(Rr/Rs)^2) - ((mur-1)/mur) * ((Rm/Rs)^2 - (Rr/Rm)^2)")
            print(f"denominator = {denominator:.8f}")
            
            # Final result
            Ba[i] = (mu0 * M / mur) * numerator / denominator
            print(f"Ba[{i}] = (mu0*M/mur) * numerator / denominator")
            print(f"Ba[{i}] = ({mu0:.2e} * {M:.2f} / {mur:.3f}) * {numerator:.8f} / {denominator:.8f}")
            print(f"Ba[{i}] = {Ba[i]:.8f}")
            print()
            
        else:  # Higher harmonics (i>1 in MATLAB)
            if i < 3:  # Only show first few for brevity
                print(f"For i={i+1} (n={n}, np={np}):")
                print("MATLAB Formula:")
                print("Ba(i)=2*(mu0*M/mur)*(np/(np^2-1))*((Rs/Rm)^(np+1))*((np-1)+2*(Rr/Rm)^(np+1)-(np+1)*(Rr/Rm)^(2*np))/(((mur+1)/mur)*(1-(Rr/Rs)^(2*np))-((mur-1)/mur)*((Rm/Rs)^(2*np)-(Rr/Rm)^(2*np)));")
                print()
                print("Python equivalent:")
                
                # Factors
                factor1 = 2 * (mu0 * M / mur)
                factor2 = np / (np**2 - 1)
                factor3 = (Rs/Rm)**(np + 1)
                
                print(f"factor1 = 2 * (mu0*M/mur) = 2 * ({mu0:.2e} * {M:.2f} / {mur:.3f}) = {factor1:.8f}")
                print(f"factor2 = np/(np^2-1) = {np}/({np}^2-1) = {np}/{np**2-1} = {factor2:.8f}")
                print(f"factor3 = (Rs/Rm)^(np+1) = ({Rs/Rm:.6f})^{np+1} = {factor3:.8f}")
                
                # Numerator
                numerator = (np - 1) + 2*(Rr/Rm)**(np + 1) - (np + 1)*(Rr/Rm)**(2*np)
                print(f"numerator = (np-1) + 2*(Rr/Rm)^(np+1) - (np+1)*(Rr/Rm)^(2*np)")
                print(f"numerator = ({np}-1) + 2*({Rr/Rm:.6f})^{np+1} - ({np}+1)*({Rr/Rm:.6f})^{2*np}")
                print(f"numerator = {numerator:.8f}")
                
                # Denominator
                denominator = ((mur+1)/mur) * (1-(Rr/Rs)**(2*np)) - ((mur-1)/mur) * ((Rm/Rs)**(2*np) - (Rr/Rm)**(2*np))
                print(f"denominator = ((mur+1)/mur)*(1-(Rr/Rs)^(2*np)) - ((mur-1)/mur)*((Rm/Rs)^(2*np)-(Rr/Rm)^(2*np))")
                print(f"denominator = {denominator:.8f}")
                
                # Final result
                Ba[i] = factor1 * factor2 * factor3 * numerator / denominator
                print(f"Ba[{i}] = factor1 * factor2 * factor3 * numerator / denominator")
                print(f"Ba[{i}] = {factor1:.8f} * {factor2:.8f} * {factor3:.8f} * {numerator:.8f} / {denominator:.8f}")
                print(f"Ba[{i}] = {Ba[i]:.8f}")
                print()
    
    return Ba

def boundary_value_problem_matlab_format():
    """
    Boundary value problem for magnet region in MATLAB format
    """
    print("Boundary Value Problem for Magnet Region:")
    print("=========================================")
    print()
    print("For n>1 harmonics in magnet region:")
    print("Matrix equation: [A11 A12] [An] = [B1]")
    print("                 [A21 A22] [Bn]   [B2]")
    print()
    print("Where:")
    print("A11 = rm_ratio^(np-1)")
    print("A12 = rm_ratio^(-np-1)")  
    print("A21 = rr_ratio^(np-1)")
    print("A22 = rr_ratio^(-np-1)")
    print()
    print("B1 = -source_term * rm_ratio^(np+1)  # From airgap matching")
    print("B2 = 0  # Zero field at rotor core")
    print()
    print("source_term = mu0 * Mn / (mur * (np^2 - 1))")
    print()
    print("Solution:")
    print("det = A11*A22 - A12*A21")
    print("numerator_An = B1*A22 - B2*A12")
    print("numerator_Bn = A11*B2 - A21*B1")
    print("An = numerator_An / det")
    print("Bn = numerator_Bn / det")
    print()
    print("Complete solution inside magnet:")
    print("Br_magnet = [An * r_ratio^(np-1) + Bn * r_ratio^(-np-1) + particular] * cos(np * theta)")
    print("where particular = source_term * (r_ratio^(np+1) - rm_ratio^(np+1))")

if __name__ == "__main__":
    # Example to show the format
    print("This file demonstrates the Python formulas written in MATLAB format")
    print("for easy verification against your MATLAB code.")
    print()
    boundary_value_problem_matlab_format()