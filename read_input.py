"""
Module for reading DALTON input  
The functions of this module is:
read_mol_quad(filename, nm)
read_magnet(filename, nm)
def read_polari(filename, nm)
def read_spinrot(filename, natom, nm)
read_nucquad(filename, natom, nm)
def read_nucquad(filename, natom, nm)
def read_optrot(filename, nm)
def read_2d_input(filename, nm)
def read_3d_input(filename, nm)
read_4d_input(filename, natom, nm)
read_quartic_force_field(filename, n_cord)
read_DALTON_values_4d_reduced
read_DALTON_values_4d_full
read_DALTON_values_3d_reduced(filename)
read_DALTON_values_3d_full(filename)
read_cubic_force_field(filename, n_cord)
read_cubic_force_field_chiral(filename, n_cord)
write_to_file(molecule, property_type, results, n_atom = None)
"""

from numpy import array, zeros, add, subtract, diag, double
import numpy as np
import re
import pydoc

def read_molecule(filename): 
    """Reads the MOLECULE.INP file generated be DALTON, and extracts the 
    information we need from it.
    
    filename: The name of the MOLECULE.INP file as saved in the input
              directory
    return: The cartessian coordinates of the molecule as a list
            The masses of the atoms as a list
            The number of atoms as a list
            The charge of the atoms as a list
            The number of atoms in the molecule as an int
            The atoms making up the molecule as a list
    """
    
    f = open(filename, 'r')
    charge_list = []
    num_atoms_list = []
    atom_list =[]
    atomicmass1 = {'O': 15.9994, 'H': 1.00794, 'C':12.0107, 'D':2.013553212724,'T':3.0160492, 'F':18.998403}
    
    coordinates = [] # contains the [x,y,z] coordinates of the input atoms
    mass = []   # contains the corresponding masses of the atoms

    finished = 0

    while (finished == 0): 
        mline = re.search('(?<=ypes\=)\w+',f.readline())

        if mline:
            atomtypes = int(mline.group(0)) #The input file specifies how many different types of atoms are present
            finished = 1
    while(atomtypes > 0):
        mline = f.readline()
        charge = int(re.search('(?<=arge\=)\w+',mline).group(0))
        num_atoms = int(re.search('(?<=toms\=)\w+',mline).group(0))
        charge_list.append(charge)
        num_atoms_list.append(num_atoms)

        for i in range(num_atoms):
            mline = f.readline().split()
            atom_list.append(mline[0])
            mass.append(atomicmass1[mline[0]])
            coordinates.append(mline[1:4])

        atomtypes -= 1

    f.close()
    coordinates = array(coordinates, double)

    return coordinates, mass, num_atoms_list, charge_list, sum(num_atoms_list),  atom_list

def read_hessian(filename, n_coords): 
    """Reads a hessian from file.
    
    filename: The name of the file the hessian is contained in. 
    n_coords: The number of cartessian coordinates needed to express
              the location of the molecule ie. 3 * number of atoms 
    return: A 2-dimensional np.array containg the hessian
    """
    dummy = []
    f = open(filename, 'r')
    hessian = zeros((n_coords, n_coords))

    #dummy = f.readline()
    for i in range(n_coords):
        a = f.readline().split()
        for j in range(n_coords):
            hessian[i,j] = float(a[j])
    f.close()

    return hessian
    
def read_MOLQUAD(filename, nm):
    """Imports things needed from the DALTON.OUT that I need"""
    
    f = open(filename, 'r')
    second_deriv = zeros((nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('Molecular quadrupole moment second derivatives',cur_line):
            finished = 1
            

    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
            
        second_deriv[mode][0][0]= mline[1]
        second_deriv[mode][0][1]= mline[2]  
        second_deriv[mode][1][1]= mline[3]
        second_deriv[mode][0][2]= mline[4]  
        second_deriv[mode][1][2]= mline[5]  
        second_deriv[mode][2][2]= mline[6]   
        
    #second_deriv_t = second_deriv[mode].transpose()
    #second_deriv_temp = add(second_deriv[mode], second_deriv_t) 
    #second_deriv[mode] = subtract(second_deriv_temp , diag(second_deriv[mode].diagonal()))   
    
    f.close()
    return second_deriv
    
def read_MAGNET(filename, nm):
    
    f = open(filename, 'r')
    second_deriv_magnet = zeros((nm,3,3))
    second_deriv_g = zeros((nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('Magnetizability tensor second derivatives',cur_line):
            finished = 1
    
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
     
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
        second_deriv_magnet[mode][0][0]= mline[1]
        second_deriv_magnet[mode][0][1]= mline[2]  
        second_deriv_magnet[mode][1][1]= mline[3]
              
        second_deriv_magnet[mode][0][2]= mline[4]  
        second_deriv_magnet[mode][1][2]= mline[5]  
        second_deriv_magnet[mode][2][2]= mline[6]
        
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline() 
    dummy = f.readline()
    dummy = f.readline()
     
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
        second_deriv_g[mode][0][0]= mline[1]
        second_deriv_g[mode][0][1]= mline[2]  
        second_deriv_g[mode][1][1]= mline[3]
        second_deriv_g[mode][0][2]= mline[4]  
        second_deriv_g[mode][1][2]= mline[5]  
        second_deriv_g[mode][2][2]= mline[6]
        
        second_deriv_t = second_deriv_g[mode].transpose()
        second_deriv_temp = add(second_deriv_g[mode], second_deriv_t) 
        second_deriv_g[mode] = subtract(second_deriv_temp, diag(second_deriv_g[mode].diagonal())) 
        
    f.close()
    
    return second_deriv_magnet 

def read_GFACTOR(filename, nm):
    
    f = open(filename, 'r')
    second_deriv_magnet = zeros((nm,3,3))
    second_deriv_g = zeros((nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('Magnetizability tensor second derivatives',cur_line):
            finished = 1
    
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
     
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
        second_deriv_magnet[mode][0][0]= mline[1]
        second_deriv_magnet[mode][0][1]= mline[2]  
        second_deriv_magnet[mode][1][1]= mline[3]
              
        second_deriv_magnet[mode][0][2]= mline[4]  
        second_deriv_magnet[mode][1][2]= mline[5]  
        second_deriv_magnet[mode][2][2]= mline[6]
        
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline() 
    dummy = f.readline()
    dummy = f.readline()
     
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
        second_deriv_g[mode][0][0]= mline[1]
        second_deriv_g[mode][0][1]= mline[2]  
        second_deriv_g[mode][1][1]= mline[3]
        second_deriv_g[mode][0][2]= mline[4]  
        second_deriv_g[mode][1][2]= mline[5]  
        second_deriv_g[mode][2][2]= mline[6]
        
        second_deriv_t = second_deriv_g[mode].transpose()
        second_deriv_temp = add(second_deriv_g[mode], second_deriv_t) 
        second_deriv_g[mode] = subtract(second_deriv_temp, diag(second_deriv_g[mode].diagonal())) 
        
    f.close()
    
    return second_deriv_g
    
def read_POLARI(filename, nm):
    """Imports things needed from the DALTON.OUT that I need"""
    
    f = open(filename, 'r')
    second_deriv = zeros((nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('ty second derivatives',cur_line):
            line_split = cur_line.split()
            property_type = line_split[0] + line_split[1]
            finished = 1
            

    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
            
        second_deriv[mode][0][0]= mline[1]
        second_deriv[mode][0][1]= mline[2]  
        second_deriv[mode][1][1]= mline[3]
        second_deriv[mode][0][2]= mline[4]  
        second_deriv[mode][1][2]= mline[5]  
        second_deriv[mode][2][2]= mline[6]   
        
    f.close()
    return second_deriv

def read_SPINROT(filename, natom, nm):
    """Imports things needed from the DALTON.OUT that I need"""
    
    f = open(filename, 'r')
    second_deriv = zeros((natom,nm,3,3))
    property_type = 0
    dummy = []
    values = zeros((9))
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('second derivatives for',cur_line):
            line_split = cur_line.split()
            property_type = line_split[0] + line_split[1]
            finished = 1
            
    for atom in range(natom):

        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
            
        if atom != 0:
            dummy = f.readline()
            dummy = f.readline()
                                
        for mode in range(nm):
            mline = f.readline()
            mline = mline.replace('********', '0.000') #This is just a theory
            mline = mline.replace(' ', '')
            
            index1 = 1 #Before the decimal place
            index2 = 0 #Which value we are at
                
            for i in range(len(mline)):                
                if (mline[i] == '.'):
                    values[index2] = mline[index1:i+4]
                    index2 = index2 + 1
                    index1 = i+4
            second_deriv[atom][mode][0][0]= values[0]
            second_deriv[atom][mode][0][1]= values[1] 
            second_deriv[atom][mode][0][2]= values[2] 
            second_deriv[atom][mode][1][0]= values[3]
            second_deriv[atom][mode][1][1]= values[4]  
            second_deriv[atom][mode][1][2]= values[5]
            second_deriv[atom][mode][2][0]= values[6]
            second_deriv[atom][mode][2][1]= values[7]  
            second_deriv[atom][mode][2][2]= values[8] 

    f.close()
    return second_deriv, property_type
      
def read_NUCQUAD(filename, natom, nm):
    
    """Imports things needed from the DALTON.OUT that I need"""
    
    f = open(filename, 'r')
    second_deriv = zeros((natom,nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('second derivatives for:',cur_line):
            line_split = cur_line.split()
            property_type = line_split[0] + line_split[1]
            finished = 1
            
    for atom in range(natom):

        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
            
        if atom != 0:
            dummy = f.readline()
            dummy = f.readline()
                                
        for mode in range(nm):
            mline = f.readline()
            mline = mline.split()
            
            second_deriv[atom][mode][0][0]= mline[1]
            second_deriv[atom][mode][0][1]= mline[2] 
            second_deriv[atom][mode][0][2]= mline[3] 
            second_deriv[atom][mode][1][1]= mline[4]  
            second_deriv[atom][mode][1][2]= mline[5]  
            second_deriv[atom][mode][2][2]= mline[6]

    f.close()
    return second_deriv, property_type

def read_OPTROT(filename, nm):
    
    f = open(filename, 'r')
    second_deriv_optrot = zeros((nm,3,3))
    dummy = []
    
    finished = 0
    
    while (finished != 3):
        cur_line = f.readline()
        if re.search('second derivatives',cur_line):
            finished = finished + 1
    
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
     
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
        second_deriv_optrot[mode][0][0]= mline[1]
        second_deriv_optrot[mode][0][1]= mline[2]  
        second_deriv_optrot[mode][1][1]= mline[3]
              
        second_deriv_optrot[mode][0][2]= mline[4]  
        second_deriv_optrot[mode][1][2]= mline[5]  
        second_deriv_optrot[mode][2][2]= mline[6]
        
    f.close()
    return second_deriv_optrot
        
def read_2d_input(filename, nm):
    
    f = open(filename, 'r')
    second_deriv = zeros((6,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('second derivatives',cur_line):
            finished = 1    
    
    dummy = f.readline()
    dummy = f.readline()
    
    for mode in range(nm):
        mline = f.readline()
        mline = mline.split()
        second_deriv[mode][0] = mline[1]
        second_deriv[mode][1] = mline[2]
        second_deriv[mode][2] = mline[3]
    
    f.close()
    return second_deriv    

def read_SHIELD(filename, natom, nm):
    """Imports things needed from the DALTON.OUT that I need"""
    
    f = open(filename, 'r')
    second_deriv = zeros((natom,nm,3,3))
    property_type = 0
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('second derivatives for:',cur_line):
            line_split = cur_line.split()
            property_type = line_split[0] + line_split[1]
            finished = 1
            
    for atom in range(natom):

        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
            
        if atom != 0:
            dummy = f.readline()
            dummy = f.readline()
                                
        for mode in range(nm):
            mline = f.readline()
            mline = mline.split()
            
            second_deriv[atom][mode][0][0]= mline[1]
            second_deriv[atom][mode][0][1]= mline[2] 
            second_deriv[atom][mode][0][2]= mline[3] 
            second_deriv[atom][mode][1][0]= mline[4]
            second_deriv[atom][mode][1][1]= mline[5]  
            second_deriv[atom][mode][1][2]= mline[6]
            second_deriv[atom][mode][2][0]= mline[7]
            second_deriv[atom][mode][2][1]= mline[8]  
            second_deriv[atom][mode][2][2]= mline[9]

    f.close()
    return second_deriv, property_type
    
def read_quartic_force_field(filename, n_cord):
    """Imports the quartic force field from DALTON"""
    
    f = open(filename, 'r')
    quartic_force_field = zeros((n_cord, n_cord, n_cord, n_cord))
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('Numerical fourth derivative of energy in symmetry coordinates',cur_line):
            finished = 1
    
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    dummy = f.readline()
    
    for D4 in range(n_cord):
        
        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
        for D3 in range(n_cord):
            dummy = f.readline()
            dummy = f.readline()
            for D2 in range(n_cord):
                mline = f.readline()
                values = zeros((n_cord))
                mline = mline.replace(' ', '')
                
                index1 = 0 #Before the decimal place
                index2 = 0 #Which value we are at
                
                for i in range(len(mline)):                
                    if (mline[i] == '.'):
                        values[index2] = mline[index1:i+7]
                        index2 = index2 + 1
                        index1 = i+7
                    
                for D1 in range(6):
                    quartic_force_field[D4][D3][D2][D1] = values[D1]
        
            dummy = f.readline()
            
            for D2 in range(n_cord):
                mline = f.readline()
                values = zeros((n_cord))
                mline = mline.replace(' ', '')
                
                index1 = 0 #Before the decimal place
                index2 = 0 #Which value we are at
                
                for i in range(len(mline)):                
                    if (mline[i] == '.'):
                        values[index2] = mline[index1:i+7]
                        index2 = index2 + 1
                        index1 = i+7

                for D1 in range(6):
                    quartic_force_field[D4][D3][D2][D1 + 6] = values[D1]
    f.close()
    
    return quartic_force_field
                
def read_DALTON_NUCQUAD(filename, natom):
    """ For testing purposes, extracts the correct values from DALTON"""
    
    f = open(filename, 'r')
    
    uncorrected_values = zeros((natom,3,3))
    corrections = zeros((natom,3,3))
    corrected_values = zeros((natom,3,3))
    
    dummy = []
    
    finished = 0
    while (finished != 2):
        cur_line = f.readline()
        if not cur_line: raise Exception("Dalton file does not contain the values looked for")
        if re.search('Vibrationally corrected',cur_line):
            finished = finished + 1
            
    for atom in range(natom):

        dummy = f.readline()
        dummy = f.readline()
            
        if atom != 0:
            dummy = f.readline()
            dummy = f.readline()
            dummy = f.readline()
            dummy = f.readline()
            dummy = f.readline()
                                           
        
        mline = f.readline()
        mline = mline.split()
        
        uncorrected_values[atom][0][0] = mline[1]
        corrections[atom][0][0] = mline[2]
        corrected_values[atom][0][0] = mline[3]
        
        mline = f.readline()
        mline = mline.split()
        uncorrected_values[atom][0][1] = mline[1]
        corrections[atom][0][1] = mline[2]
        corrected_values[atom][0][1] = mline[3]
            
        mline = f.readline()
        mline = mline.split()
        uncorrected_values[atom][1][1] = mline[1]
        corrections[atom][1][1] = mline[2]
        corrected_values[atom][1][1] = mline[3]
        
        mline = f.readline()
        mline = mline.split()
        uncorrected_values[atom][0][2] = mline[1]
        corrections[atom][0][2] = mline[2]
        corrected_values[atom][0][2] = mline[3]
        
        mline = f.readline()
        mline = mline.split()
        uncorrected_values[atom][1][2] = mline[1]
        corrections[atom][1][2] = mline[2]
        corrected_values[atom][1][2] = mline[3]
        
        mline = f.readline()
        mline = mline.split()
        uncorrected_values[atom][2][2] = mline[1]
        corrections[atom][2][2] = mline[2]
        corrected_values[atom][2][2] = mline[3]
            
    return uncorrected_values, corrections, corrected_values
 
def read_DALTON_SHIELD(filename, natom):
    """ For testing purposes, extracts the correct values from DALTON"""
    
    f = open(filename, 'r')
    
    uncorrected_values = zeros((natom,3,3))
    corrections = zeros((natom,3,3))
    corrected_values = zeros((natom,3,3))
    
    dummy = []
    
    finished = 0
    while (finished != 2):
        cur_line = f.readline()
        if not cur_line: raise Exception("Dalton file does not contain the values looked for")
        if re.search('Vibrationally corrected',cur_line):
            finished = finished + 1
            
    for atom in range(natom):

        dummy = f.readline()
        dummy = f.readline()
            
        if atom != 0:
            dummy = f.readline()
            dummy = f.readline()
            dummy = f.readline()
                                           
        for i in range(3):
            mline = f.readline()
            mline = mline.split()
        
            uncorrected_values[atom][0][i] = mline[1]
            corrections[atom][0][i] = mline[2]
            corrected_values[atom][0][i] = mline[3]
        
            mline = f.readline()
            mline = mline.split()
            uncorrected_values[atom][1][i] = mline[1]
            corrections[atom][1][i] = mline[2]
            corrected_values[atom][1][i] = mline[3]
            
            mline = f.readline()
            mline = mline.split()
            uncorrected_values[atom][2][i] = mline[1]
            corrections[atom][2][i] = mline[2]
            corrected_values[atom][2][i] = mline[3]
            
    return uncorrected_values, corrections, corrected_values   

def read_DALTON_SPINROT(filename, natom):
    """ For testing purposes, extracts the correct values from DALTON"""
    
    f = open(filename, 'r')
    
    uncorrected_values = zeros((natom,3,3))
    corrections = zeros((natom,3,3))
    corrected_values = zeros((natom,3,3))
    
    dummy = []
    
    finished = 0
    while (finished != 2):
        cur_line = f.readline()
        if not cur_line: raise Exception("Dalton file does not contain the values looked for")
        if re.search('Vibrationally corrected',cur_line):
            finished = finished + 1
            
    for atom in range(natom):

        dummy = f.readline()
        dummy = f.readline()
            
        if atom != 0:
            dummy = f.readline()
            dummy = f.readline()
            dummy = f.readline()
                                           
        for i in range(3):
            mline = f.readline()
            mline = mline.split()
        
            uncorrected_values[atom][0][i] = mline[1]
            corrections[atom][0][i] = mline[2]
            corrected_values[atom][0][i] = mline[3]
        
            mline = f.readline()
            mline = mline.split()
            uncorrected_values[atom][1][i] = mline[1]
            corrections[atom][1][i] = mline[2]
            corrected_values[atom][1][i] = mline[3]
            
            mline = f.readline()
            mline = mline.split()
            uncorrected_values[atom][2][i] = mline[1]
            corrections[atom][2][i] = mline[2]
            corrected_values[atom][2][i] = mline[3]
            
    return uncorrected_values, corrections, corrected_values  
    
def read_DALTON_MAGNET(filename):
    """ For testing purposes, extracts the correct values from DALTON"""
    
    f = open(filename, 'r')
    
    uncorrected_values = zeros((3,3))
    corrections = zeros((3,3))
    corrected_values = zeros((3,3))
    
    dummy = []
    
    finished = 0
    while (finished != 2):
        cur_line = f.readline()
        if not cur_line: raise Exception("Dalton file does not contain the values looked for")
        if re.search('Vibrationally corrected',cur_line):
            finished = finished + 1

    dummy = f.readline()
    dummy = f.readline()
    
    
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[0][0] = mline[1]
    corrections[0][0] = mline[2]
    corrected_values[0][0] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[0][1] = mline[1]
    corrections[0][1] = mline[2]
    corrected_values[0][1] = mline[3]
            
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[0][2] = mline[1]
    corrections[0][2] = mline[2]
    corrected_values[0][2] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[1][1] = mline[1]
    corrections[1][1] = mline[2]
    corrected_values[1][1] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[1][2] = mline[1]
    corrections[1][2] = mline[2]
    corrected_values[1][2] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[2][2] = mline[1]
    corrections[2][2] = mline[2]
    corrected_values[2][2] = mline[3]
            
    return uncorrected_values, corrections, corrected_values
    
def read_DALTON_OPTROT(filename):
    """ For testing purposes, extracts the correct values from DALTON"""
    
    f = open(filename, 'r')
    
    uncorrected_values = zeros((3,3))
    corrections = zeros((3,3))
    corrected_values = zeros((3,3))
    
    dummy = []
    
    finished = 0
    while (finished != 2):
        cur_line = f.readline()
        if not cur_line: raise Exception("Dalton file does not contain the values looked for")
        if re.search('Vibrationally corrected',cur_line):
            finished = finished + 1

    dummy = f.readline()
    dummy = f.readline()
    
    
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[0][0] = mline[1]
    corrections[0][0] = mline[2]
    corrected_values[0][0] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[0][1] = mline[1]
    corrections[0][1] = mline[2]
    corrected_values[0][1] = mline[3]
            
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[0][2] = mline[1]
    corrections[0][2] = mline[2]
    corrected_values[0][2] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[1][1] = mline[1]
    corrections[1][1] = mline[2]
    corrected_values[1][1] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[1][2] = mline[1]
    corrections[1][2] = mline[2]
    corrected_values[1][2] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[2][2] = mline[1]
    corrections[2][2] = mline[2]
    corrected_values[2][2] = mline[3]
            
    return uncorrected_values, corrections, corrected_values

def read_DALTON_MOLQUAD(filename):
    """ For testing purposes, extracts the correct values from DALTON"""
    
    f = open(filename, 'r')
    
    uncorrected_values = zeros((3,3))
    corrections = zeros((3,3))
    corrected_values = zeros((3,3))
    
    dummy = []
    
    finished = 0
    while (finished != 2):
        cur_line = f.readline()
        if not cur_line: raise Exception("Dalton file does not contain the values looked for")
        if re.search('Vibrationally corrected',cur_line):
            finished = finished + 1

    dummy = f.readline()
    dummy = f.readline()
    
    
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[0][0] = mline[1]
    corrections[0][0] = mline[2]
    corrected_values[0][0] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[0][1] = mline[1]
    corrections[0][1] = mline[2]
    corrected_values[0][1] = mline[3]
            
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[0][2] = mline[1]
    corrections[0][2] = mline[2]
    corrected_values[0][2] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[1][1] = mline[1]
    corrections[1][1] = mline[2]
    corrected_values[1][1] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[1][2] = mline[1]
    corrections[1][2] = mline[2]
    corrected_values[1][2] = mline[3]
        
    mline = f.readline()
    mline = mline.split()
    uncorrected_values[2][2] = mline[1]
    corrections[2][2] = mline[2]
    corrected_values[2][2] = mline[3]
            
    return uncorrected_values, corrections, corrected_values
    
def read_DALTON_GFACTOR(filename):
    """ For testing purposes, extracts the correct values from DALTON"""
    
    f = open(filename, 'r')
    
    uncorrected_values = zeros((3,3))
    corrections = zeros((3,3))
    corrected_values = zeros((3,3))
    
    dummy = []
    
    finished = 0
    while (finished != 3):
        cur_line = f.readline()
        if not cur_line: raise Exception("Dalton file does not contain the values looked for")
        if re.search('Vibrationally corrected',cur_line):
            finished = finished + 1

    dummy = f.readline()
    dummy = f.readline()

                                           
    for i in range(3):
        
        mline = f.readline()
        mline = mline.split()
        
        uncorrected_values[0][i] = mline[1]
        corrections[0][i] = mline[2]
        corrected_values[0][i] = mline[3]
        
        mline = f.readline()
        mline = mline.split()
        uncorrected_values[1][i] = mline[1]
        corrections[1][i] = mline[2]
        corrected_values[1][i] = mline[3]
            
        mline = f.readline()
        mline = mline.split()
        uncorrected_values[2][i] = mline[1]
        corrections[2][i] = mline[2]
        corrected_values[2][i] = mline[3]
            
    return uncorrected_values, corrections, corrected_values   

def read_DALTON_values_2d(filename):
    
    f = open(filename, 'r')
    
    uncorrected_values = zeros((3))
    corrections = zeros((3))
    corrected_values = zeros((3))
    dummy = []
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if not cur_line: raise Exception("Dalton file does not contain the values looked for")
        if re.search('Vibrationally corrected',cur_line):
            finished = finished + 1

    dummy = f.readline()
    dummy = f.readline()
    
    for i in range(3):
        mline = f.readline()
        mline = mline.split()        
        uncorrected_values[i] = mline[1]
        corrections[i] = mline[2]
        corrected_values[i] = mline[3]
        
    return uncorrected_values, corrections, corrected_values        

def read_cubic_force_field(filename, n_cord): #Make this one generic
    """Imports the quartic force field from DALTON"""
    
    f = open(filename, 'r')
    cubic_force_field = zeros((n_cord, n_cord, n_cord))
    dummy = []
    counter = n_cord # Make an if counting, and thus reading correctly..
    
    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('Anharmonic force constants',cur_line):
            finished = 1

    for D3 in range(n_cord):
        
        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
        
        if(D3 != 0):
            dummy = f.readline()
            dummy = f.readline()
            dummy = f.readline()
        
        for D2 in range(n_cord):
            mline = f.readline()
            mline = mline.split()
            for D1 in range(5):
                cubic_force_field[D3][D2][D1] = mline[D1 + 1]
        
        dummy = f.readline()
        dummy = f.readline()
            
        for D2 in range(n_cord):
            mline = f.readline()
            mline = mline.split()
            for D1 in range(5):
                cubic_force_field[D3][D2][D1 + 5] = mline[D1 + 1]
                
        dummy = f.readline()
        dummy = f.readline()
            
        for D2 in range(n_cord):
            mline = f.readline()
            mline = mline.split()
            for D1 in range(2):
                cubic_force_field[D3][D2][D1 + 10] = mline[D1 + 1]
    f.close()
    return cubic_force_field

def read_cubic_force_field_h2o(filename, n_coords):
    """
    Reads the cubic force field calculated by DALTON from file.
    
    filename: The name of the file the cubic force field is contained in 
    n_coords: The number of cartessian coordinates needed to express
              the location of the molecule ie. 3 * number of atoms 
    return: A 3-dimensional np.array containg the cubic force field
    """
    dummy = []
    cubic_force_field = zeros((n_coords, n_coords, n_coords))
    f = open(filename, 'r')

    for i in range(n_coords):
        dummy = f.readline()
        for j in range(n_coords):
            a = f.readline().split()
            for k in range(n_coords):
                cubic_force_field[i,j,k] = float(a[k])
    f.close()

    return cubic_force_field
    
def read_eigenvector(filename, n_atoms):
    
    n_nm = 3*n_atoms - 6
    eigenvector = zeros((12, 6))

    f = open(filename, 'r')
    
    for i in range(12):
        a = f.readline().split()
        for j in range(6):
            eigenvector[i,j] = float(a[j])
    
    f.close()
    return eigenvector
        
def read_cubic_force_field_chiral(filename, n_cord): #Make this one generic
    """Imports the chiral cubic force field from DALTON"""
      
    f = open(filename, 'r')
    cubic_force_field = zeros((n_cord, n_cord, n_cord))
    dummy = []
    counter = n_cord # Make an if counting, and thus reading correctly..

    finished = 0
    while (finished == 0):
        cur_line = f.readline()
        if re.search('Anharmonic force constants',cur_line):
            finished = 1

    for D3 in range(n_cord):
        
        dummy = f.readline()
        dummy = f.readline()
        dummy = f.readline()
        
        if(D3 != 0):
            dummy = f.readline()
            dummy = f.readline()
            dummy = f.readline()
        
        for D2 in range(n_cord):
            mline = f.readline()
            mline = mline.split()
            for D1 in range(5):
                cubic_force_field[D3][D2][D1] = mline[D1 + 1]
        
        dummy = f.readline()
        dummy = f.readline()
            
        for D2 in range(n_cord):
            mline = f.readline()
            mline = mline.split()
            for D1 in range(5):
                cubic_force_field[D3][D2][D1 + 5] = mline[D1 + 1]
                
        dummy = f.readline()
        dummy = f.readline()
            
        for D2 in range(n_cord):
            mline = f.readline()
            mline = mline.split()
            for D1 in range(5):
                cubic_force_field[D3][D2][D1 + 10] = mline[D1 + 1]
    f.close()
    return cubic_force_field

