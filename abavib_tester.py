
from numpy import array, dot, sqrt, set_printoptions, reshape, multiply, divide, add, subtract, diag
import numpy as np
from scipy import mat, linalg, double
from read_input import *
from abavib import *

dipole = mat([[0.083445,0.020685,-0.008117] 
,[-0.123263,-0.080377,0.031540]
,[-0.366672,-0.483183,0.189596]])

dipole_pre = mat([ 0.37370174, 0.49133014, -0.19279329])


def main():
    mol_name = 'MOLECULE.INP'
    hessian_name = 'hessian'
    hessian_vib_name = 'hessian_vibprop'
    cff_name = 'cff'
    coordinates, masses,  num_atoms_list, charge_list, n_atoms = read_molecule(mol_name)

    n_coords = 3 * n_atoms
    n_nm = n_coords - 6 
    
    hessian = read_hessian(hessian_name, n_coords)
    hessian_t = hessian.transpose()
    hessian_temp = add(hessian, hessian_t) 
    hessian = subtract(hessian_temp , diag(hessian.diagonal()))
    
    eig, eigvec_reduced, freq, eigvec = fundamental_freq(hessian, num_atoms_list, charge_list, coordinates, n_atoms)
    
    cubic_force_field = read_cubic_force_field(cff_name, n_coords)
  
    cff_norm, cff_norm_reduced = to_normal_coordinates_3D(cubic_force_field, eigvec, n_atoms)
   
    effective_geometry_norm = effective_geometry(cff_norm_reduced, freq, n_atoms)
    
    effective_geometry_cart = to_cartessian_coordinates(effective_geometry_norm, n_atoms, eigvec_reduced)
    
    print "effective geometry"
    print effective_geometry_norm
    
    #dipole_moment_diff, dipole_moment_corrected = get_dipole_moment(dipole, n_nm, eig, dipole_pre, True)
   
    #shield = get_4D_property("Shield", shield_deriv, n_nm, n_atoms, EVAL, True)
    #spin_r = get_4D_property("Spin - Rotation Constant", spin_r_deriv, n_nm, n_atoms, EVAL, True)
    
    #nuc_quad_deriv, prop_type = read_4d_input("property", 4, 6)
    #nuc_quad = get_4D_property(prop_type, nuc_quad_deriv, n_nm, n_atoms, EVAL, True)
    
    #spin_spin_deriv, prop_type = read_3d_input("SPIN-SPIN", 6)
    #spin_spin = get_3D_property(spin_spin_deriv, n_nm, EVAL)
    
    mol_quad_deriv, prop_type, pre_property = read_mol_quad("MOLQUAD", 3)
    
    #print "Pre Property"
    #print pre_property
    
    #print "mol_quad_deriv"
    #print mol_quad_deriv
    
    print "The eigenvalues"
    print eig
    
    print "The Frequencies"
    print freq
    
    print "effective geometry"
    print effective_geometry_cart 
    
    mol_quad = get_3D_property(prop_type, mol_quad_deriv, n_nm, eig, True)
    
    #polari_deriv, prop_type = read_polari("POLARI", 6)
    #polari = get_3D_property(prop_type, polari_deriv, n_nm, EVAL, True)  
    
    #quartic_force_field = read_quartic_force_field('quartic',12)     
    
   
    
set_printoptions(suppress=True) #Avoid scientific notation when printing arrs

main()

