"""
Copyright (c) 2013-2014 Benedicte Ofstad
Distributed under the GNU Lesser General Public License v3.0. 
For full terms see the file LICENSE.md.
"""

import read_input as ri
from numpy import array, zeros, vstack, dot, identity, sqrt, set_printoptions, compress, reshape, multiply, divide, add, subtract, diag, absolute, sort, argsort, fliplr
from numpy import vectorize, diff, copy, transpose, inner, diagonal
import numpy as np
import re # regular expressions
import os
from scipy import mat, linalg, double
import pydoc
import pdb

"""
Module for computing vibrational molecular properties. All variables 
needed to compute the relevant properties are also calculated in this 
module

The parameters needed in this modeule is:
filename = The filename of where a hessian is saved
filename = the filename of where the MOLECULE.INP is saved
filename = the filename of where the cubic force field is saved
pre_property = the second derivative of the property vibrational
               averaging is to be conducted one
uncorrected_property = the property before the vibrational averaging
correct_propert = The property after the vibrational averaging, for 
                  testing purposes
"""

class Molecule:
    
    def __init__(self, name):
    
        self.name = name
        self.input_name = self.get_input_name()
        self.linear = 0
        self.n_atoms = ri.read_molecule(self.get_molecule_input_name())[4]
        self.coordinates = ri.read_molecule(self.get_molecule_input_name())[0]
        self.atom_list = ri.read_molecule(self.get_molecule_input_name())[5]
        self.n_coordinates = self.get_coordinates() 
        self.number_of_normal_modes = self.get_number_of_normal_modes()
        self.hessian = self.get_hessian()
        
        self.eigenvalues,\
        self.eigenvectors,\
        self.frequencies, \
        self.eigenvalues_full, \
        self.eigenvectors_full = self.diagonalize(self.hessian)
        
        self.cff_norm = self.to_normal_coordinates_3D\
        (ri.read_cubic_force_field_anal(self.get_cubic_force_field_name(), self.n_coordinates))
        
        #self.qff_norm = self.to_normal_coordinates_4D(ri.read_quartic_force_field1(self.input_name + 'quartic', self.n_coordinates))
        
        self.effective_geometry = self.get_effective_geometry()
        
        open(self.get_output_name(), 'w').close() # As we are appending to the output, the old results must be deleted before each run
    
    def get_input_name(self):
        """Returns the name of the directory the input files are to be 
        read are found."""
        input_name = "input_" + self.name + "/"
        return input_name
        
    def get_output_name(self):
        """Returns the name of the directory the output information is to
        be saved in."""
        output_file_name = "output/" + self.name 
        return output_file_name

    def get_cubic_force_field_name1(self):
        """Returns the  name of the file containing the cubic force field
         of the molecule"""
        input_name = self.get_input_name()
        cff_name = self.input_name + 'effective geometry'
        return cff_name
        
    def get_cubic_force_field_name(self):
        """Returns the  name of the file containing the cubic force field
         of the molecule"""
        input_name = self.get_input_name()
        cff_name = self.input_name + 'cubic_force_field'
        return cff_name
        
    def get_eigenvector_name(self):
        """Returns the  name of the file containing the cubic force field
         of the molecule"""
        input_name = self.get_input_name()
        cff_name = self.input_name + 'hess'
        return cff_name

    def get_molecule_input_name(self):
        """Returns the  name of the file containing the geometry of the
         molecule in cartessian coordinates"""
        input_name = self.get_input_name()
        molecule_input_name = self.input_name + 'MOLECULE.INP'
        return molecule_input_name
    
        """Returns the  name of the file containing the cubic force field
         of the molecule"""
        input_name = self.get_input_name()
        cff_name = self.input_name + 'cubic_force_field'
        return cff_name
        
    def get_coordinates(self):
        """Returns the number of coordinates of the molecule"""
        n_coordinates = self.n_atoms * 3
        return n_coordinates
        
    def get_number_of_normal_modes(self):
        """Returns the number of normal modes of the molecule"""
        number_of_normal_modes = self.n_coordinates - 6
        
        if(self.name == "hf"):
            number_of_normal_modes = self.n_coordinates - 5
            
        return number_of_normal_modes
        
    def get_hessian(self):
        """Returns the hessian of the molecule"""
        hessian_name = self.get_input_name() + 'hessian'
        hessian = ri.read_hessian(hessian_name, self.n_atoms*3)
        return hessian
    
    def hessian_trans_rot(self, hessian, cart_coord, nr_normal_modes, n_atoms): 
        """Projects the hessian of the molecule so that it can be used to 
        determine the normal coordinates of the molecule. This projected
        hessian is referred to as the analystical hessian in DALTON

        hessian: The hessian of the molecule as an np.array
        nr_normal_modes: The number of normal modes of the molecule as an 
                     int
        n_atoms: The number of atoms the molecule consists of
        return: The projected hessian as a 2 dimensional matrix                  
        """
        
        trans1 = [1,0,0]
        trans2 = [0,1,0]
        trans3 = [0,0,1]

        trans_rot = zeros((3* n_atoms, 6))

        for atom in range(n_atoms):
            ij = atom*3
            trans_rot[ij, 0] = 1.0
            trans_rot[ij + 1, 1] = 1.0
            trans_rot[ij + 2, 2] = 1.0
            trans_rot[ij, 3] = -1* cart_coord[atom, 1]
            trans_rot[ij + 1, 3] = cart_coord[atom, 0]
            trans_rot[ij + 1, 4] = -1* cart_coord[atom, 2]
            trans_rot[ij + 2, 4] = cart_coord[atom, 1]
            trans_rot[ij, 5] = cart_coord[atom, 2]
            trans_rot[ij + 2, 5] = -1* cart_coord[atom, 0]
            ij = ij + 3
        
        trans_rot = linalg.qr(mat(trans_rot), mode = 'economic') [0:1]
        
        trans_rot = -1* mat(trans_rot[0])
        trans_rot_proj = -(trans_rot * (trans_rot.T) - mat(identity(3* n_atoms)))  
        #print(trans_rot)
        trans_rot_proj = mat(trans_rot_proj)
        
        hess_proj = (trans_rot_proj * mat(hessian)) * trans_rot_proj 
        #print(hess_proj)
        hess_proj = mat(hess_proj)
        
        if(self.n_atoms == 2):
            hess_proj = zeros((6 , 6))
            hess_proj[0,0] = 0.5
            hess_proj[0,3] = 0.5
            hess_proj[1,1] = 1
            hess_proj[2,2] = 1
            hess_proj[3,3] = 0.5
            hess_proj[4,4] = 1
            hess_proj[5,5] = 1

        return hess_proj
                                   
    def mass_hessian(self, masses):
        """
        Creates an np.array where the masses of the different atoms in
        the molecule are placed at the diagonal. This is a help function
        used to make a mass weighted hessian. 
        masses: The masses of the atoms in the molecule, can be recieved
                from the read_molecule() function.
        returns: The masses alond the diagonal of a 2 dimensional np.array
        """
        
        masses = array(masses)
        m = zeros(3*len(masses))
        m_e = 1822.8884796 # conversion factor from a.m.u to a.u 
        identity_matrix = identity(3*len(masses))
        
        index = 0
        for i in range(len(masses)):
            m[index] = masses[i]
            index = index +1
            m[index] = masses[i]
            index = index +1
            m[index] = masses[i]
            index = index +1
            
        m = m * m_e
        m = 1/(sqrt(m))
        
        M = identity_matrix*m
        
        return M
            
    def masswt_hessian(self, num_atoms_list, charge_list): 
        """
        Converts the hessian into the mass weighted hessian
        num_atoms_list: A list of the number of atoms of each type
        charge list: A list of the charges of each atom 
        returns the mass weighted hessian (np.array)"""
        atomicmass = {9.0:18.998403, 8.0: 15.994915, 1.0: 1.007825}
        
        deuterium = 2.014102
        tritinum = 3.016
        
        m_e = 1822.8884796 # conversion factor from a.m.u to a.u 

        M = zeros((3*sum(num_atoms_list),3*sum(num_atoms_list)))
        s = 0

        for i in range(len(num_atoms_list)):
            for j in 3*range(num_atoms_list[i]):
                M[s, s] = 1/(sqrt(atomicmass[charge_list[i]]*m_e))
                s += 1
        return M
        
    def diagonalize(self, hessian): 
        """Computes the normal coordinates, eigenvalues and fundamental 
        frequencies of the molecule. 
        hessian: The hessian of the molecule as an np.array
        num_atoms_list: A list of the how many of each atoms type there are
        charge list: A list over the charges of the atoms in the molecle
        coordinates: The cartessian coordinates of the molecule as an np.array
        n_atoms: The number of atoms composing the molecule as an int
        masses: The masses of the atoms in a molecule along the diagonal of
                an np.array
        returns: The non-zero eigenvalues of the molecule as an np.array
                 The normal coordinates of the molecule, ie. the eigenvectors
                 corresponding the non-zero eigevalues as an np.array
                 The fundalmental frequencies of the molecule as an np.array
                 All the eigenvectos of the molecules, both the ones 
                 corresponding the zero and non-zero eigenvalues as an np.array
        """
       
        coordinates, masses, num_atoms_list, charge_list,\
        n_atoms, atom_list = ri.read_molecule(self.get_molecule_input_name())
    
        
        M_I = self.mass_hessian(masses)
        if (self.linear):
            n_nm += 1
        
        hess_proj = self.hessian_trans_rot(hessian, coordinates, self.number_of_normal_modes, n_atoms)
        
        hessian_proj = dot(M_I.transpose(), hess_proj)
        hessian_proj = dot(hessian_proj, M_I)
        
        
        v, La = linalg.eig(hessian_proj)
    
        v_reduced = v[:self.number_of_normal_modes]
         
        v_args = v_reduced.argsort()[::-1]
        v_reduced = array(v_reduced, double)
        v_reduced = v_reduced[v_args]
        
        La= dot(M_I, array(La, double))
        La_reduced = La[:,v_args]
    
        freq = sqrt(absolute(v_reduced))
        
        closefunc = lambda x: abs(x) < 0.0001
        closevect = vectorize(closefunc)
        reldiff = lambda x, y: x/y
        reldiff = vectorize(reldiff)
        eqsign = lambda x, y: x*y > 0
        eqsign = vectorize(eqsign)

        return v_reduced, La_reduced, freq, v, La
        
    def to_normal_coordinates_3D(self, cubic_force_field):
        """Converts a cubic force field represented by cartessian coordinates
        into a cubic force field represented by normal coordinates
        
        cubic_force_field: A 3 dimenesional np.array of the cubic force field
                           represented in cartessian coordinates
        self.eigenvectors_full: The eigenvectors of the molecule corresponding to the non-
                zero eigenvalues, can be attained for fundamental_freq() (np.array)
        n_atoms: The number of atoms constituting the molecule as an int
        returns: The cubic force field in normal coordinates (3D np.array)
        """
        
        
        
        cubic_force_field_clone = copy(cubic_force_field)
        cff_norm = zeros((self.n_coordinates,self.n_coordinates,self.n_coordinates))
        
        #In matrix operations
        
        #cubic_force_field_clone = transpose(cubic_force_field_clone)
        
        #cubic_force_field_clone = dot(cubic_force_field_clone, self.eigenvectors_full)
        #cubic_force_field_clone = transpose(cubic_force_field_clone,(0,2,1))
        
        #cubic_force_field_clone = dot(cubic_force_field_clone,self.eigenvectors_full)
        #cubic_force_field_clone = transpose(cubic_force_field_clone,(2,1,0))
        
        #eigenvectors_full_clone = self.eigenvectors_full[:,:self.number_of_normal_modes]
        #cubic_force_field_clone = cubic_force_field_clone[:self.number_of_normal_modes,:self.number_of_normal_modes,:]
        
        #cubic_force_field_clone = dot(cubic_force_field_clone, eigenvectors_full_clone)
        #cff_norm = transpose(cubic_force_field_clone, (1,0,2))
        
        #for i in range(self.n_coordinates):
        #    for j in range(self.n_coordinates):
        #        for k in range(self.n_coordinates):
        #            temp = 0
        #            for ip in range(self.number_of_normal_modes):
        #                for jp in range(self.number_of_normal_modes):
        #                    for kp in range(self.number_of_normal_modes):
        #                        temp = temp + cubic_force_field[ip,jp,kp]* self.eigenvectors_full[kp,k]*self.eigenvectors_full[jp,j]*self.eigenvectors_full[ip,i]
        #            cff_norm[k,j,i]= temp 
        
        
        # Fortran like implementation:
        
        for i in range(self.n_coordinates):
            for j in range(self.n_coordinates):
                for k in range(self.n_coordinates):
                    temp = 0
                    for kp in range(self.n_coordinates):
                        temp = temp + cubic_force_field_clone[kp,j,i]* self.eigenvectors_full[kp,k]
                    cff_norm[k,j,i]= temp
                    
        for i in range(self.n_coordinates):
            for j in range(self.n_coordinates):
                for k in range(self.n_coordinates):
                    temp = 0
                    for jp in range(self.n_coordinates):
                        temp = temp + cff_norm[k,jp,i]* self.eigenvectors_full[jp,j]
                    cubic_force_field_clone[k,j,i]= temp
                    
        for i in range(self.number_of_normal_modes):
            for j in range(self.number_of_normal_modes):
                for k in range(self.number_of_normal_modes):
                    temp = 0
                    for ip in range(self.n_coordinates):
                        temp = temp + cubic_force_field_clone[k,j,ip]* self.eigenvectors_full[ip, i]
                    cff_norm[k,j,i]= temp
        return cff_norm 

    def to_normal_coordinates_4D(self, quartic_force_field):
        """Converts a quartic force field represented by cartessina coordinates
        into a quartic force field represented by normal coordinates
        
        quartic_force_field: A 4 dimenesional np.array of the quartic force field
                           represented in cartessian coordinates
        self.eigenvectors_full: The eigenvectors of the molecule corresponding to the non-
                zero eigenvalues, can be attained from fundamental_freq() (np.array)
        n_atoms: The number of atoms constituting the molecule as an int
        returns: The quartic force field in normal coordinates ( 4d np.array)
        """
        qff_norm = zeros((self.n_coordinates, self.n_coordinates, self.n_coordinates, self.n_coordinates)) 
        qff_norm1 = zeros((self.n_coordinates, self.n_coordinates, self.n_coordinates, self.n_coordinates)) 
        quartic_force_field_clone = copy(quartic_force_field)
        quartic_force_field_clone1 = copy(quartic_force_field)
         
        #This is an exact implementation of the equation. But this is slow
        
        #for i in range(self.n_coordinates):
        #    for j in range(self.n_coordinates):
        #        for k in range(self.n_coordinates):
        #            for l in range(self.n_coordinates):
        #                temp = 0
        #                for ip in range(self.number_of_normal_modes):
        #                    for jp in range(self.number_of_normal_modes):
        #                        for kp in range(self.number_of_normal_modes):
        #                            for lp in range(self.number_of_normal_modes):
        #                                temp = temp + quartic_force_field_clone1[ip,jp,kp,lp]*self.eigenvectors_full[lp,l]*self.eigenvectors_full[kp,k]*self.eigenvectors_full[jp,j]*self.eigenvectors_full[ip,i]
        #                qff_norm1[l,k,j,i]= temp
                        
        
        for i in range(self.n_coordinates):
            for j in range(self.n_coordinates):
                for k in range(self.n_coordinates):
                    for l in range(self.n_coordinates):
                        temp = 0
                        for lp in range(self.number_of_normal_modes):
                            temp = temp + quartic_force_field_clone[lp,k,j,i]* self.eigenvectors_full[lp,l]
                        qff_norm[l,k,j,i]= temp
        for i in range(self.n_coordinates):
            for j in range(self.n_coordinates):
                for k in range(self.n_coordinates):
                    for l in range(self.n_coordinates):
                        temp = 0
                        for kp in range(self.number_of_normal_modes):
                            temp = temp + qff_norm[l,kp,j,i]* self.eigenvectors_full[kp,k]
                        qff_norm[l,k,j,i]= temp
        for i in range(self.n_coordinates):
            for j in range(self.n_coordinates):
                for k in range(self.n_coordinates):
                    for l in range(self.n_coordinates):
                        temp = 0
                        for jp in range(self.number_of_normal_modes):
                            temp = temp + qff_norm[l,k,jp,i]* self.eigenvectors_full[jp,j]
                        qff_norm[l,k,j,i]= temp
        for i in range(self.n_coordinates):
            for j in range(self.n_coordinates):
                for k in range(self.n_coordinates):
                    for l in range(self.n_coordinates):
                        temp = 0
                        for ip in range(self.number_of_normal_modes):
                            temp = temp + qff_norm[l,k,j,ip]* self.eigenvectors_full[ip,i]
                        qff_norm[l,k,j,i]= temp
        
        #quartic_force_field_clone = copy(quartic_force_field)
        #quartic_force_field_clone = transpose(quartic_force_field_clone)
        
        #quartic_force_field_clone = dot(quartic_force_field_clone, self.eigenvectors_full)
        #quartic_force_field_clone = transpose(quartic_force_field_clone,(0,3,2,1))
        
        #quartic_force_field_clone = dot(quartic_force_field_clone,self.eigenvectors_full)
        #quartic_force_field_clone = transpose(quartic_force_field_clone,(3,2,1,0))
        
        #quartic_force_field_clone = dot(quartic_force_field_clone, self.eigenvectors_full)
        #quartic_force_field_clone = transpose(quartic_force_field_clone,(2,1,0,3))
        
        #self.eigenvectors_full = self.eigenvectors_full[:self.number_of_normal_modes,:self.number_of_normal_modes]
        #quartic_force_field_clone = quartic_force_field_clone\
        #    [:self.number_of_normal_modes,:self.number_of_normal_modes,:self.number_of_normal_modes,:self.number_of_normal_modes]
        
        #quartic_force_field_clone = dot(quartic_force_field_clone,self.eigenvectors_full)
        #qff_norm = transpose(quartic_force_field_clone, (1,0,3,2))
    
        return qff_norm 
        
    def get_effective_geometry(self):
        """Converts normal coordinates into cartessian coordinates.
        
        normal coordinates: The normal coordinates of which are to be converted
        n_atoms: The number of atoms constituting the molecule as an int
        eigvec: The eigenvectors of the molecule corresponging to the non
                zero eigenvalues, can be attained for fundamental_freq() (np.array)
        returns: cartessian coordinates as an np.array.
        """
        factor = sqrt(1822.8884796) #I DONT KNOW WHY?!
        #anstrom_to_bohr = 1.88971616463
        
        print(self.eigenvectors)
         
        effective_geometry_norm = self.effective_geometry_norm(self.cff_norm)
        cartessian_coordinates = np.sum(factor*effective_geometry_norm*self.eigenvectors, 1)
        
        #instead of reshape() this will fail if it cannot be done efficiently:
        cartessian_coordinates.shape = (self.n_atoms, 3) 
        
        effective_geometry = self.coordinates + cartessian_coordinates
        
        print(self.coordinates)
       
        print("corrections")
        print(cartessian_coordinates)
        print ("effective geometry")
        print (effective_geometry)
        
        return effective_geometry

    def effective_geometry_norm(self, cff_norm):
        """Computes the effective geometry of a molecule.

        cff_norm: The cubic force field of the molecule in normal coordinates
                  as an np.array
        frequencies: The fundamental frequencies of the molecule as an np.array
        n_atoms: The number of atoms constituting the molecule as an int
        return: The effective geometry in normal coordinates as an np.arrays
        """
        # 1822.8884796 is the conversion factor from a.m.u to a.u
        
        
        prefix = -1/(4*sqrt(1822.8884796)*self.frequencies**2)
        
        molecular_geometry = np.sum(divide(cff_norm.diagonal(0,0,1)\
        [:self.number_of_normal_modes,:self.number_of_normal_modes],self.frequencies), axis=1)
        molecular_geometry = molecular_geometry*prefix
        
        #DALTON Implementation
        #molecular_geometry = zeros((self.number_of_normal_modes))
        #for i in range(self.number_of_normal_modes):
        #    prefix = -1/(4*self.frequencies[i]**2*factor)
        #    temp = 0
        #    for j in range(self.number_of_normal_modes): 
        #        temp = temp + divide(cff_norm[i,j,j], self.frequencies[j])
        #    molecular_geometry[i] = temp*prefix 
        
               
        print molecular_geometry
        return molecular_geometry 
             
